"""
Onion Price Prediction — Streamlit App
Model: onion_price_model_lr.pkl
Features: onion_price_model_features.pkl
Historical data: data/onion_features_ready.csv

Run karne ke liye:
    streamlit run onion_price_app.py
"""

import sys
from datetime import timedelta
from pathlib import Path

import pandas as pd
import streamlit as st

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from backend import (
    DATE_COL,
    VARIETY_FLAG_COL,
    get_market_count,
    get_prediction_payload,
    get_top_markets,
    load_data,
)

st.set_page_config(
    page_title="Onion Price Predictor",
    page_icon="🧅",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Onion Price Predictor")
st.write("Predict onion prices using the trained machine-learning model.")

simple_data = load_data()
simple_variety = st.selectbox("Onion variety", ["Red", "Other"])
simple_variety_value = 1 if simple_variety == "Red" else 0

# Anchor date: the last date we actually HAVE data for, for this variety.
# All forecasting counts forward from here, not from "today" on the calendar —
# if your data collection is a few days behind, the anchor will be a few days
# behind too, and that gap is baked into every forecast below it.
variety_rows = simple_data[simple_data[VARIETY_FLAG_COL] == simple_variety_value]
anchor_date = variety_rows[DATE_COL].max() if not variety_rows.empty else None

st.caption(
    f"Data mein is variety ka aakhri known date: **{anchor_date.date()}**"
    if anchor_date is not None
    else "Is variety ke liye koi data nahi mila."
)

forecast_mode = st.radio(
    "Forecast kis tarah chahiye?",
    ["Din count se (jaise agle 7 din)", "Specific date se (jaise 01/01/2027)"],
    horizontal=True,
)

simple_days = None
target_date = None

if forecast_mode.startswith("Din count"):
    simple_days = st.slider("Forecast days", min_value=1, max_value=30, value=7)
else:
    if anchor_date is not None:
        default_target = (anchor_date + timedelta(days=30)).date()
        min_allowed = (anchor_date + timedelta(days=1)).date()
        max_allowed = (anchor_date + timedelta(days=1095)).date()  # 3-year hard cap
        target_date = st.date_input(
            "Kaunsi date ka price chahiye?",
            value=default_target,
            min_value=min_allowed,
            max_value=max_allowed,
        )
        days_ahead = (target_date - anchor_date.date()).days

        if days_ahead > 30:
            st.warning(
                f"⚠️ Yeh date data ke aakhri known din se **{days_ahead} din** door hai. "
                "30 din se aage, prediction mostly is mahine ke **historical seasonal "
                "average** pe based hoti hai (na ki din-ba-din trend pe) — kyunki itni "
                "door ka exact number predict karna kisi bhi model ke liye reliable "
                "nahi hota. Isse ek 'typical is season mein price kaisa rehta hai' "
                "wala estimate maano, exact forecast nahi.",
                icon="⚠️",
            )
        simple_days = days_ahead

if st.button("Predict price", type="primary", disabled=(simple_days is None or simple_days <= 0)):
    with st.spinner("Calculating forecast..."):
        simple_payload = get_prediction_payload(simple_variety_value, simple_days)

    simple_forecast = simple_payload["forecast_df"]
    simple_history = simple_payload["filtered_history"]

    if simple_forecast.empty:
        st.warning(f"No data is available for the {simple_variety} variety.")
    else:
        st.subheader("Summary")
        simple_metrics = st.columns(4)
        simple_metrics[0].metric("Current price", f"₹{simple_payload['latest_price']:.2f}/kg")

        if target_date is not None:
            target_row = simple_forecast.iloc[-1]
            simple_metrics[1].metric(
                f"Price on {target_date.strftime('%d %b %Y')}",
                f"₹{target_row['predicted_price']:.2f}/kg",
            )
        else:
            simple_metrics[1].metric("Next forecast", f"₹{simple_payload['forecast_price']:.2f}/kg")

        simple_metrics[2].metric("Average forecast", f"₹{simple_payload['avg_price']:.2f}/kg")
        simple_metrics[3].metric("Active markets", get_market_count())

        # --- Reliability disclaimer -----------------------------------
        st.info(
            "📌 **Note on accuracy:** Day 1 ka forecast sabse reliable hai (real "
            "trained model se). Jaise-jaise din aage badhte hain, model gradually "
            "**seasonal pattern** (is mahine mein historically price kaisa rehta "
            "hai) pe shift karta hai, taaki lambi forecasts unrealistic (jaise "
            "zero ki taraf crash) na ho jaayein. 30+ din ki forecast ko seasonal "
            "trend samjho, din-specific exact number nahi.",
            icon="ℹ️",
        )
        if simple_days > 60:
            st.caption(
                "ℹ️ Note: 1+ saal aage ki forecast mein, alag-alag saalon ke "
                "same mahine ka price kaafi similar dikhega (jaise Feb 2025 aur "
                "Feb 2026) — kyunki model sirf har mahine ka historical seasonal "
                "pattern repeat kar raha hai, koi long-term price-growth trend "
                "assume nahi kar raha. Yeh jaan-boojh kar conservative approach "
                "hai (galat growth-guess se better)."
            )

        # --- Forecast chart — tiered by horizon so it stays readable -----
        chart_title = (
            f"{simple_variety} variety — {target_date.strftime('%d %b %Y')} tak ka path"
            if target_date is not None
            else f"{simple_variety} variety — {simple_days}-day forecast"
        )
        st.subheader(chart_title)

        chart_df = simple_forecast.copy()
        chart_df["day_number"] = range(1, len(chart_df) + 1)
        n_points = len(chart_df)

        if n_points <= 14:
            # Short forecast: one bar per day, easiest to read.
            chart_df["label"] = chart_df["date"].dt.strftime("%d %b")
            bar_display = chart_df.set_index("label")[["predicted_price"]]
            bar_display.columns = ["Predicted price (₹/kg)"]
            st.bar_chart(bar_display, y_label="Price (₹/kg)")

        elif n_points <= 90:
            # Medium forecast: too many days to show one-by-one, so we
            # average into weeks. Far fewer points, still shows the trend.
            weekly = (
                chart_df.set_index("date")["predicted_price"]
                .resample("W")
                .mean()
                .reset_index()
            )
            weekly["label"] = weekly["date"].dt.strftime("%d %b '%y")
            line_display = weekly.set_index("label")[["predicted_price"]]
            line_display.columns = ["Predicted price (₹/kg)"]
            st.line_chart(line_display, y_label="Price (₹/kg)")
            st.caption("Har point ek hafte ka average price hai, roz-roz ka nahi.")

        else:
            # Long forecast (months/years out): daily numbers this far out
            # aren't meaningful anyway (see disclaimer above — it's mostly
            # the seasonal pattern), so we show a monthly average instead.
            # This is fewer bars, includes the year in the label so months
            # from different years never look identical, and honestly
            # reflects what the model is actually telling you at this range.
            monthly = (
                chart_df.set_index("date")["predicted_price"]
                .resample("ME")
                .mean()
                .reset_index()
            )
            monthly["label"] = monthly["date"].dt.strftime("%b %Y")
            bar_display = monthly.set_index("label")[["predicted_price"]]
            bar_display.columns = ["Avg predicted price (₹/kg)"]
            st.bar_chart(bar_display, y_label="Price (₹/kg)")
            st.caption(
                "Har bar ek mahine ka average price hai. Itni door ki forecast "
                "mostly seasonal pattern pe based hoti hai, isliye monthly view "
                "hi sabse honest tarika hai isse dikhane ka — daily numbers is "
                "range mein misleading precision denge."
            )

        with st.expander("Uncertainty range dekho (advanced, daily detail)"):
            st.caption(
                "Yeh dikhata hai forecast kitna 'off' ho sakta hai — jitna aage "
                "ki date, utni chaudi range. Yahan har din alag se dikhaya gaya "
                "hai (year ke saath), isliye lambi forecasts ke liye scroll karna "
                "pad sakta hai."
            )
            chart_df["label"] = chart_df["date"].dt.strftime("%d %b '%y")
            # Cap uncertainty at ±50% — beyond that the number stops being
            # a useful range and just looks broken (e.g. a negative lower bound).
            uncertainty_pct = (chart_df["day_number"] * 0.02).clip(upper=0.5)
            chart_df["lower_bound"] = chart_df["predicted_price"] * (1 - uncertainty_pct)
            chart_df["upper_bound"] = chart_df["predicted_price"] * (1 + uncertainty_pct)
            band_display = chart_df.set_index("label")[["lower_bound", "predicted_price", "upper_bound"]]
            band_display.columns = ["Lower estimate", "Predicted price", "Upper estimate"]
            st.line_chart(band_display, y_label="Price (₹/kg)")

        with st.expander("Poora din-ba-din forecast dekho"):
            st.dataframe(
                simple_forecast.rename(
                    columns={"date": "Date", "predicted_price": "Predicted price (₹/kg)"}
                ),
                use_container_width=True,
                hide_index=True,
            )

        if not simple_history.empty:
            st.subheader("Recent actual prices")
            simple_history_chart = simple_history[[DATE_COL, "modal_price"]].rename(
                columns={"modal_price": "Price (₹/kg)"}
            )
            st.line_chart(simple_history_chart.set_index(DATE_COL))

        st.subheader("Top markets")
        simple_markets = get_top_markets(10).rename(columns={"price_per_kg": "Price (₹/kg)"})
        st.dataframe(simple_markets, use_container_width=True, hide_index=True)

        st.download_button(
            "Download forecast CSV",
            data=simple_forecast.to_csv(index=False).encode("utf-8"),
            file_name=f"onion_{simple_variety.lower()}_forecast.csv",
            mime="text/csv",
        )

st.caption("Data source: onion_features_ready.csv")