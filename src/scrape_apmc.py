import requests
import pandas as pd
from bs4 import BeautifulSoup
import time

def scrape_apmc_onion_prices():
    """
    APMC website se onion prices scrape karo
    """
    
    base_url = "https://www.agmarknet.gov.in/SearchCommodityPrice.aspx"
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Linux; X11) AppleWebKit/537.36'
    }
    
    all_data = []
    
    try:
        print("🔄 APMC website se data scrape kar rahe hain...")
        
        # Form data
        form_data = {
            'ddlCommodity': 'Onion',
            'ddlMarket': 'Nashik',
            'ddlState': 'Maharashtra'
        }
        
        # Request
        response = requests.post(base_url, data=form_data, headers=headers, timeout=30)
        
        if response.status_code == 200:
            print(f"✅ Status: {response.status_code}")
            
            # Parse HTML
            soup = BeautifulSoup(response.content, 'html.parser')
            table = soup.find('table', {'class': 'gridview'})
            
            if table:
                rows = table.find_all('tr')
                print(f"📊 Found {len(rows)} rows")
                
                for row in rows[1:]:
                    cols = row.find_all('td')
                    if len(cols) >= 5:
                        data = {
                            'Date': cols[0].text.strip(),
                            'Market': cols[1].text.strip(),
                            'Commodity': cols[2].text.strip(),
                            'Min_Price': cols[3].text.strip(),
                            'Max_Price': cols[4].text.strip(),
                            'Avg_Price': cols[5].text.strip() if len(cols) > 5 else ''
                        }
                        all_data.append(data)
                
                time.sleep(2)
        else:
            print(f"❌ Error: {response.status_code}")
        
        # Save to CSV
        if all_data:
            df = pd.DataFrame(all_data)
            df.to_csv('data/onion_prices_scraped.csv', index=False)
            print(f"✅ {len(df)} records saved!")
            return df
        else:
            print("❌ No data found")
            return None
            
    except requests.exceptions.Timeout:
        print("❌ Timeout - server slow")
    except Exception as e:
        print(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    df = scrape_apmc_onion_prices()
    if df is not None:
        print("\n📋 First 5 rows:")
        print(df.head())

