"""
Onion Price Predictor — Backend
================================
Bridge between the trained model (onion_price_model_lr.pkl) and the
Streamlit UI (onion_price_app.py).

Data file expected at:  ../data/onion_features_ready.csv
Model files expected at: ../models/onion_price_model_lr.pkl
                          ../models/onion_price_model_features.pkl

IMPORTANT NOTES (read before trusting numbers blindly):

1. UNIT FIX: `modal_price` / `min_price` / `max_price` in the CSV are in
   Rs. per QUINTAL (standard mandi/APMC convention), not Rs. per kg.
   The UI labels everything "/kg", so this backend divides by 100 before
   returning any price to the UI. Without this, a price like 1040
   (Rs/quintal) would incorrectly show as "₹1040.00/kg" instead of the
   correct "₹10.40/kg". If your earlier screenshots showed prices like
   "₹2190.00/kg", that was almost certainly raw Rs/quintal shown without
   this conversion.

2. RECURSIVE FORECASTING: the model was trained to predict ONE day ahead
   (target column is `target_price_h1`). It was NOT trained to directly
   output a 7-day forecast. To forecast multiple days, this backend
   predicts day 1, appends that prediction to the price history, recomputes
   lag/rolling features from the updated history, predicts day 2, and so on.
   This is a standard technique but it means forecast error compounds the
   further out you go — day 7 is less reliable than day 1. Consider
   surfacing that caveat in the UI.

3. ASSUMPTIONS made because the model needs features we cannot know for
   future dates:
   - `arrivals_quintal` for future days: held constant at the last known
     actual value (we have no arrivals forecast).
   - `min_price` / `max_price` for future days: reconstructed from the
     predicted `modal_price` using the average recent price-spread ratio,
     rather than being predicted directly.
   These are reasonable defaults, not ground truth — flag clearly in the UI
   that forecasts beyond day 1 are model-estimated, not sourced data.
"""

from __future__ import annotations

import pickle
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent

DATA_PATH = PROJECT_ROOT / "data" / "onion_features_ready.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "onion_price_model_lr.pkl"
FEATURES_PATH = PROJECT_ROOT / "models" / "onion_price_model_features.pkl"

DATE_COL = "arrival_date_parsed"
PRICE_COL = "modal_price"
ARRIVALS_COL = "arrivals_quintal"
MARKET_COL = "market"
VARIETY_FLAG_COL = "variety_red"

QUINTAL_TO_KG = 100.0  # 1 quintal = 100 kg

LAGS = [1, 3, 7, 14, 30]
ROLL_WINDOWS = [7, 14, 30]


# --------------------------------------------------------------------------
# Loading (cached so we don't hit disk on every Streamlit rerun)
# --------------------------------------------------------------------------

@lru_cache(maxsize=1)
def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Data file not found at {DATA_PATH}. Check that "
            "data/onion_features_ready.csv exists in the project root."
        )
    df = pd.read_csv(DATA_PATH, parse_dates=[DATE_COL])
    df = df.sort_values(DATE_COL).reset_index(drop=True)
    return df


@lru_cache(maxsize=1)
def load_model_and_features():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
    if not FEATURES_PATH.exists():
        raise FileNotFoundError(f"Features file not found at {FEATURES_PATH}")

    model = joblib.load(MODEL_PATH)
    with open(FEATURES_PATH, "rb") as f:
        features = pickle.load(f)
    return model, features


# --------------------------------------------------------------------------
# Simple summary helpers used directly by the UI
# --------------------------------------------------------------------------

def get_market_count() -> int:
    df = load_data()
    return int(df[MARKET_COL].nunique())


def get_top_markets(n: int = 10) -> pd.DataFrame:
    """Latest known modal price per market, converted to Rs/kg, sorted desc."""
    df = load_data()
    latest = (
        df.sort_values(DATE_COL)
        .groupby(MARKET_COL, as_index=False)
        .last()[[MARKET_COL, PRICE_COL]]
    )
    latest["price_per_kg"] = latest[PRICE_COL] / QUINTAL_TO_KG
    latest = latest.drop(columns=[PRICE_COL]).rename(columns={MARKET_COL: "market"})
    return latest.sort_values("price_per_kg", ascending=False).head(n).reset_index(drop=True)


# --------------------------------------------------------------------------
# Feature engineering for a single future day, given price/arrivals history
# --------------------------------------------------------------------------

def _cyclical_month(month: int) -> tuple[float, float]:
    angle = 2 * np.pi * month / 12
    return float(np.sin(angle)), float(np.cos(angle))


def _build_feature_row(
    target_date: pd.Timestamp,
    price_history: list[float],
    arrivals_history: list[float],
    last_min_price: float,
    last_max_price: float,
    avg_spread_ratio: float,
    variety_flag: int,
) -> dict:
    """Build one row of model features for `target_date`, using price/arrivals
    history that already includes every day up to (but not including) the
    day being predicted."""

    modal_price = price_history[-1]
    arrivals_quintal = arrivals_history[-1]

    # Reconstruct min/max from the predicted modal price using the recent
    # average spread ratio (see module docstring, assumption #3).
    price_spread = modal_price * avg_spread_ratio
    min_price = last_min_price if len(price_history) == 1 else modal_price - price_spread / 2
    max_price = last_max_price if len(price_history) == 1 else modal_price + price_spread / 2

    month = target_date.month
    month_sin, month_cos = _cyclical_month(month)

    def lag(n):
        return price_history[-n - 1] if len(price_history) > n else price_history[0]

    def roll_mean(n):
        window = price_history[-n:] if len(price_history) >= n else price_history
        return float(np.mean(window))

    def roll_std(n):
        window = price_history[-n:] if len(price_history) >= n else price_history
        return float(np.std(window, ddof=1)) if len(window) > 1 else 0.0

    lag_1 = lag(1)
    lag_7 = lag(7)

    row = {
        ARRIVALS_COL: arrivals_quintal,
        "min_price": min_price,
        "max_price": max_price,
        PRICE_COL: modal_price,
        "year": target_date.year,
        "month": month,
        "day_of_week": target_date.dayofweek,
        "day_of_year": target_date.dayofyear,
        "quarter": target_date.quarter,
        "is_weekend": int(target_date.dayofweek >= 5),
        "month_sin": month_sin,
        "month_cos": month_cos,
        "modal_price_lag_1": lag_1,
        "modal_price_lag_3": lag(3),
        "modal_price_lag_7": lag_7,
        "modal_price_lag_14": lag(14),
        "modal_price_lag_30": lag(30),
        "modal_price_roll_mean_7": roll_mean(7),
        "modal_price_roll_std_7": roll_std(7),
        "modal_price_roll_mean_14": roll_mean(14),
        "modal_price_roll_std_14": roll_std(14),
        "modal_price_roll_mean_30": roll_mean(30),
        "modal_price_roll_std_30": roll_std(30),
        "price_pct_change_1d": (modal_price - lag_1) / lag_1 if lag_1 else 0.0,
        "price_pct_change_7d": (modal_price - lag_7) / lag_7 if lag_7 else 0.0,
        "price_spread": price_spread,
        "price_spread_pct": price_spread / modal_price if modal_price else 0.0,
        "arrivals_lag_1": arrivals_history[-2] if len(arrivals_history) > 1 else arrivals_history[-1],
        "arrivals_roll_mean_7": float(np.mean(arrivals_history[-7:])),
        VARIETY_FLAG_COL: variety_flag,
    }
    return row


# --------------------------------------------------------------------------
# Main entry point used by the UI
# --------------------------------------------------------------------------

def get_prediction_payload(variety_flag: int, n_days: int) -> dict:
    """Recursively forecast `n_days` ahead for the given variety.

    variety_flag: 1 for "Red", 0 for "Other" (matches the `variety_red` column).
    Returns a dict with:
        latest_price, forecast_price, avg_price   (all Rs/kg)
        forecast_df       -> columns: date, predicted_price (Rs/kg)
        filtered_history  -> columns: arrival_date_parsed, modal_price (Rs/kg)
    """
    df = load_data()
    model, feature_order = load_model_and_features()

    subset = df[df[VARIETY_FLAG_COL] == variety_flag].sort_values(DATE_COL)

    empty_result = {
        "latest_price": 0.0,
        "forecast_price": 0.0,
        "avg_price": 0.0,
        "forecast_df": pd.DataFrame(columns=["date", "predicted_price"]),
        "filtered_history": pd.DataFrame(columns=[DATE_COL, PRICE_COL]),
    }
    if subset.empty:
        return empty_result

    # Working history in Rs/quintal (native units the model was trained on)
    price_history = subset[PRICE_COL].tolist()
    arrivals_history = subset[ARRIVALS_COL].tolist()
    last_row = subset.iloc[-1]
    last_date = last_row[DATE_COL]
    last_min_price = float(last_row["min_price"])
    last_max_price = float(last_row["max_price"])

    recent_spread_ratio = (
        (subset["max_price"] - subset["min_price"]) / subset[PRICE_COL]
    ).tail(30).mean()
    if pd.isna(recent_spread_ratio):
        recent_spread_ratio = 0.3  # fallback if history is too short

    forecast_rows = []
    for step in range(1, n_days + 1):
        target_date = last_date + pd.Timedelta(days=step)
        feat_row = _build_feature_row(
            target_date=target_date,
            price_history=price_history,
            arrivals_history=arrivals_history,
            last_min_price=last_min_price,
            last_max_price=last_max_price,
            avg_spread_ratio=recent_spread_ratio,
            variety_flag=variety_flag,
        )
        X = pd.DataFrame([feat_row])[feature_order]
        predicted_price = float(model.predict(X)[0])
        predicted_price = max(predicted_price, 0.0)  # price can't go negative

        forecast_rows.append({"date": target_date, "predicted_price": predicted_price / QUINTAL_TO_KG})

        # Extend history so the next step's lag/rolling features include this prediction
        price_history.append(predicted_price)
        arrivals_history.append(arrivals_history[-1])  # carry arrivals forward flat

    forecast_df = pd.DataFrame(forecast_rows)

    payload = {
        "latest_price": float(last_row[PRICE_COL]) / QUINTAL_TO_KG,
        "forecast_price": float(forecast_df["predicted_price"].iloc[0]),
        "avg_price": float(forecast_df["predicted_price"].mean()),
        "forecast_df": forecast_df,
        "filtered_history": subset[[DATE_COL, PRICE_COL]].assign(
            **{PRICE_COL: subset[PRICE_COL] / QUINTAL_TO_KG}
        ),
    }
    return payload