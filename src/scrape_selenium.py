from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import Select
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import time
import pandas as pd

def scrape_apmc_selenium():
    """
    Selenium se APMC website scrape karo
    """
    
    print("🔄 Chrome browser open kar rahe hain...")
    
    # Chrome driver setup
    options = webdriver.ChromeOptions()
    options.add_argument('--start-maximized')
    
    driver = webdriver.Chrome(
        service=Service(ChromeDriverManager().install()),
        options=options
    )
    
    try:
        url = "https://www.agmarknet.gov.in/SearchCommodityPrice.aspx"
        print(f"📍 Opening: {url}")
        
        driver.get(url)
        
        # Page load wait kar
        time.sleep(3)
        
        print("📋 Form fill kar rahe hain...")
        
        # Commodity select kar
        try:
            commodity_select = Select(driver.find_element(By.ID, "ddlCommodity"))
            commodity_select.select_by_value("Onion")
            print("✅ Onion selected")
        except Exception as e:
            print(f"⚠️ Commodity select failed: {e}")
        
        # Market select kar
        try:
            market_select = Select(driver.find_element(By.ID, "ddlMarket"))
            market_select.select_by_visible_text("Nashik (Nashik)")
            print("✅ Nashik selected")
        except Exception as e:
            print(f"⚠️ Market select failed: {e}")
        
        # State select kar
        try:
            state_select = Select(driver.find_element(By.ID, "ddlState"))
            state_select.select_by_visible_text("MAHARASHTRA")
            print("✅ Maharashtra selected")
        except Exception as e:
            print(f"⚠️ State select failed: {e}")
        
        # Search button click kar
        try:
            search_btn = driver.find_element(By.ID, "btnSubmit")
            search_btn.click()
            print("✅ Search button clicked")
        except Exception as e:
            print(f"⚠️ Button click failed: {e}")
        
        # Results load wait kar
        time.sleep(3)
        
        # Table scrape kar
        print("📊 Results scrape kar rahe hain...")
        
        try:
            table = driver.find_element(By.CLASS_NAME, "gridview")
            rows = table.find_elements(By.TAG_NAME, "tr")
            
            all_data = []
            
            for row in rows[1:]:  # Header skip
                cols = row.find_elements(By.TAG_NAME, "td")
                
                if len(cols) >= 6:
                    data = {
                        'Date': cols[0].text.strip(),
                        'Market': cols[1].text.strip(),
                        'Commodity': cols[2].text.strip(),
                        'Min_Price': cols[3].text.strip(),
                        'Max_Price': cols[4].text.strip(),
                        'Avg_Price': cols[5].text.strip()
                    }
                    all_data.append(data)
            
            # Save to CSV
            if all_data:
                df = pd.DataFrame(all_data)
                df.to_csv('data/onion_prices_scraped.csv', index=False)
                print(f"✅ {len(df)} records saved!")
                print(df.head())
                return df
            else:
                print("❌ No data found in table")
                
        except Exception as e:
            print(f"❌ Table scrape failed: {e}")
    
    finally:
        print("🔴 Browser close kar rahe hain...")
        driver.quit()

if __name__ == "__main__":
    scrape_apmc_selenium()
