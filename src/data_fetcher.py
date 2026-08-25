import requests
import pandas as pd
import os
import time
import csv
from dotenv import load_dotenv

load_dotenv()


class DataFetcher:
    BASE_URL = "https://api.data.gov.in/resource"

    def __init__(self):
        self.api_key = os.getenv("DATA_GOV_API_KEY")
        self.resource_id = os.getenv("RESOURCE_ID")
        if not self.api_key:
            raise ValueError("API key missing — .env file check kar")
        if not self.resource_id:
            raise ValueError("Resource ID missing — .env file check kar")

    def fetch_batch(self, commodity="Onion", limit=50, offset=0, retries=3):
        """Chhota batch fetch karta hai (limit=50 default, curl-tested fast size)."""
        url = f"{self.BASE_URL}/{self.resource_id}"
        params = {
            "api-key": self.api_key,
            "format": "json",
            "limit": limit,
            "offset": offset,
            "filters[commodity]": commodity,
        }

        for attempt in range(1, retries + 1):
            try:
                response = requests.get(url, params=params, timeout=20)
                response.raise_for_status()
                data = response.json()
                records = data.get("records", [])
                return pd.DataFrame(records)
            except requests.exceptions.Timeout:
                print(f"  Timeout attempt {attempt}/{retries} (offset={offset})")
                time.sleep(2)
            except requests.exceptions.RequestException as e:
                print(f"  Error attempt {attempt}/{retries}: {e}")
                time.sleep(2)

        return pd.DataFrame()

    def fetch_all_incremental(self, commodity="Onion", batch_size=50, max_records=10000,
                                output_file="data/raw_onion_all_india.csv"):
        """Chhote batches me fetch karta hai aur HAR batch ke baad CSV me turant save karta hai."""
        offset = 0
        first_batch = True
        total_saved = 0

        while offset < max_records:
            df = self.fetch_batch(commodity=commodity, limit=batch_size, offset=offset)

            if df.empty:
                print(f"Offset {offset}: koi record nahi mila. Stopping.")
                break

            # Pehli baar header ke saath likho, uske baad append karo
            mode = "w" if first_batch else "a"
            header = first_batch
            df.to_csv(output_file, mode=mode, header=header, index=False)
            first_batch = False

            total_saved += len(df)
            print(f"Offset {offset}: {len(df)} records saved (total so far: {total_saved})")

            offset += batch_size
            time.sleep(0.5)  # server pe thoda rest, rate-limit se bachne ke liye

            if len(df) < batch_size:
                print("Last page reach ho gaya (kam records mile).")
                break

        return total_saved

    def filter_local(self, df, state=None, market=None):
        if df.empty:
            return df
        result = df.copy()
        if state and "State" in result.columns:
            result = result[result["State"].str.strip().str.lower() == state.lower()]
        if market and "Market" in result.columns:
            result = result[result["Market"].str.strip().str.lower() == market.lower()]
        return result.reset_index(drop=True)


if __name__ == "__main__":
    fetcher = DataFetcher()

    print("Fetching Onion data in SMALL batches (India-wide)...")
    print("Ye thoda time lega kyunki chhote-chhote requests ja rahe hain, patience rakh.\n")

    total = fetcher.fetch_all_incremental(
        commodity="Onion",
        batch_size=50,
        max_records=5000,  # pehle chhota target rakha hai testing ke liye
        output_file="data/raw_onion_all_india.csv",
    )

    print(f"\nTotal records fetched: {total}")

    if total > 0:
        df_all = pd.read_csv("data/raw_onion_all_india.csv")

        df_maharashtra = fetcher.filter_local(df_all, state="Maharashtra")
        print(f"Maharashtra records: {len(df_maharashtra)}")
        if not df_maharashtra.empty:
            df_maharashtra.to_csv("data/raw_onion_maharashtra.csv", index=False)

        df_lasalgaon = fetcher.filter_local(df_all, state="Maharashtra", market="Lasalgaon")
        print(f"Lasalgaon records: {len(df_lasalgaon)}")
        if not df_lasalgaon.empty:
            df_lasalgaon.to_csv("data/raw_onion_lasalgaon.csv", index=False)
            print("Saved to data/raw_onion_lasalgaon.csv")
        else:
            print("Lasalgaon abhi nahi mila is chhote sample me — zyada records fetch karne padenge.")
    else:
        print("Kuch bhi fetch nahi hua — API abhi bhi down/slow lag rahi hai.")
