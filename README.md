# 🧅 Lasalgaon Onion Price Predictor

A machine learning web application that forecasts onion prices for the **Lasalgaon APMC market** (Maharashtra, India), built to help farmers, traders, and buyers make informed decisions using historical price trends.



---

## 📌 Overview

Onion prices in India are highly volatile and vary significantly across seasons and markets. This project uses historical price data from the **Lasalgaon APMC** — Asia's largest onion market — to train a predictive model that forecasts future prices, presented through an interactive web app.

---

## 📊 Data

- **Source:** [Kaggle](https://www.kaggle.com/) — a dataset containing onion price records across multiple markets in India
- **Processing:** Records specific to the Lasalgaon APMC market were extracted, cleaned, and prepared for model training

---

## ⚙️ Workflow

| Step | Description |
|------|-------------|
| 1. Data Collection | Extracted Lasalgaon-specific records from a Kaggle India onion price dataset |
| 2. Feature Engineering | Cleaned and engineered relevant features from raw price data |
| 3. Model Training | Trained the prediction model on **Google Colab** |
| 4. Model Export | Serialized the trained model as a `.pkl` file |
| 5. Project Structuring | Organized the codebase into a clean folder structure using Sublime Text |
| 6. Deployment & Testing | Ran and tested the application locally via **Ubuntu (WSL)** |

---

## 📁 Project Structure

onion-price-predictor/
├── data/ # Raw and processed datasets
├── models/ # Trained model files (.pkl)
├── notebooks/ # Jupyter/Colab notebooks
├── src/
│ ├── onion_price_app.py # Main Streamlit app
│ ├── data_fetcher.py # Data loading utilities
│ └── scrape_apmc.py # APMC data scraper
├── requirements.txt
└── README.md


## 🛠️ Tech Stack

- **Language:** Python
- **ML/Data:** Pandas, Scikit-learn
- **Web App:** Streamlit
- **Training Environment:** Google Colab
- **Development Environment:** Sublime Text, WSL (Ubuntu)


---

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- pip
- numpy
- pandas
- scikit-learn
  

### Installation

```bash
git clone https://github.com/rohitshewale4165-eng/lasalgaon-onion-price-predictor.git
cd lasalgaon-onion-price-predictor
pip install -r requirements.txt
```

### Run the App

```bash
streamlit run src/onion_price_app.py
```

The app will open in your browser at `http://localhost:8501`

---

## 📈 Future Improvements

- Add support for more APMC markets across Maharashtra
- Improve model accuracy with additional weather/seasonal features
- Deploy the app on Streamlit Cloud for public access

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

## 🙋‍♂️ Author

**Rohit Shewale**

---
