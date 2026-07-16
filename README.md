# Investment Support Agent (ISA)

A Machine Learning-powered Investment Support Agent built with Python and Streamlit. This project was developed as an assignment to predict stock returns and validate those predictions against real-world data.

---

## 📌 Assignment Context & Liberties Taken

The core objective of this assignment was to select a few stocks, train an agent to predict the next 3 months, and then validate those predictions against real market data.

**Liberty Taken on Training Window:**  
The originally it was suggested to use 3 months of historical data for training. However, 3 months (approx. 60 trading days) is fundamentally insufficient for training robust Machine Learning models, especially when attempting to predict 30, 60, or 90 days into the future (due to extreme target overlap and data starvation). 

To build a more scientifically sound ML pipeline, **I took the liberty of expanding the training window to 6 months (October 2025 – March 2026)**. The prediction simulation then assumes today is April 1, 2026, and validates against the actual following 3 months (April, May, June 2026).

---

## 📊 Project Specifications & Assumptions

### 1. Data Source
- **Yahoo Finance (yfinance):** All historical OHLCV (Open, High, Low, Close, Volume) data is fetched programmatically using the `yfinance` Python library. No manual CSV downloads are required.

### 2. Supported Sectors & Companies
To demonstrate sector-based trends, the agent covers 16 companies split across 4 sectors:
- **IT:** TCS, Infosys, HCLTech, Wipro
- **Banking:** HDFC Bank, ICICI Bank, SBI, Axis Bank
- **Pharma:** Sun Pharma, Cipla, Divi's Labs, Dr. Reddy's
- **Automobile:** Hero MotoCo, Maruti Suzuki, Mahindra & Mahindra, Bajaj Auto

### 3. Machine Learning Pipeline & Assumptions
- **Target Variable:** The models predict **forward percentage return** (not raw stock price). This normalises the data across companies of vastly different share prices.
- **Features:** 17 technical indicators are engineered from raw OHLCV data, including Moving Averages (MA5, MA10, MA20), EMA, RSI, MACD, Bollinger Bands, and Rolling Volatility.
- **Algorithms:** The pipeline trains `LinearRegression` (with StandardScaler), `RandomForestRegressor`, and `XGBoost`. 
- **Hyperparameter Tuning & Regularisation:** Tree depths are intentionally constrained (max_depth=3) to prevent the models from overfitting the small 6-month dataset.
- **Model Selection:** For every Sector × Horizon combination, all 3 algorithms are trained, and the one with the highest R² score on a validation split is automatically saved as the champion model.

### 4. Validation Strategy
- The model trains exclusively on data up to **March 31, 2026**.
- The "Predict" tab simulates predictions as if today is **April 1, 2026**.
- The "Validate" tab goes back to Yahoo Finance, downloads the *actual* stock prices for April, May, and June 2026, and directly compares them against the model's predictions, calculating MAE, RMSE, and Error %.
### 5. Model Limitations & Data Constraints
**Important Note on Accuracy:** While the original prompt suggested 3 months of data (which was expanded to 6 months for this project), this is still mathematically insufficient to capture true market cycles. In the real world, stock markets are driven by unpredictable macroeconomic shifts, news events, and impulsive market sentiment that technical indicators alone cannot foresee over short horizons. 

Because the ML models only have 6 months of historical context (amounting to ~125 trading days), they will occasionally struggle to predict massive 15-20% short-term rallies or black swan crashes. The models are designed to identify conservative, technical trends, and during periods of extreme volatility, the Error % margin on the Validate tab will naturally increase. 

---
## 🚀 Setup Instructions (from ZIP)

If you have received this project as a ZIP file, follow these steps to run it locally:

### 1. Extract and Navigate
Extract the ZIP file and open your terminal/command prompt inside the extracted folder:
```bash
cd investment-support-agent
```

### 2. Create a Virtual Environment (Recommended)
```bash
python -m venv venv

# On Windows:
venv\Scripts\activate
# On macOS / Linux:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Gemini API Key (Optional but Recommended)
The app features an AI-explanation module powered by Google Gemini.
1. Create a file named `.env` in the root folder.
2. Add your API key:
   ```env
   GEMINI_API_KEY=your_google_gemini_api_key_here
   ```
*(Note: If no key is provided, or if the free-tier quota is exceeded, the app will still function perfectly, it will simply skip generating the text explanations).*

### 5. Run the Training Pipeline (Optional if pre-trained)
**Note:** If your ZIP file already includes the `models/` and `data/` folders, you can skip this step entirely and go straight to Step 6!

If you want to train the models from scratch yourself, this script downloads the data, engineers features, trains all models, and saves the `.joblib` files.
```bash
python train.py
```
*(Expected runtime: 2–5 minutes depending on internet connection).*

### 6. Launch the Application
```bash
streamlit run ui/app.py
```
Open `http://localhost:8501` in your browser.

---

## 💻 Using the Application

1. **Predict Tab:** Enter a hypothetical budget, select a sector, and select a horizon (30/60/90 days). Click Generate. The UI will provide predicted returns, confidence scores, and two allocation strategies (All-in vs Diversified Risk-adjusted).
2. **Validate Tab:** After generating a prediction, switch to the Validate tab. Click the Validate button to pull the real-world data from the future (Apr-Jun 2026) and see how accurate the ML models were against reality!
