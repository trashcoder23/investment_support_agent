# Investment Support Agent (ISA)

A production-grade, Autonomous AI Investment Analyst built with **LangGraph**, **Mistral**, and **Streamlit**. 

This project was engineered to replace static numeric machine learning models with a dynamic, reasoning-capable AI Agent that synthesizes both structured financial metrics (Yahoo Finance) and unstructured market sentiment (live news APIs).

---

## 📌 Architecture Overview

The core of this system is a Directed Acyclic Graph (DAG) state machine managed by **LangGraph**. When a user submits a natural language query (e.g., *"I have 50000 INR. Should I buy TCS?"*), the Agent autonomously routes the query through the following pipeline:

1. **Intent Parsing Node:** Uses NLP to extract the target company, ticker, financial budget, and prediction horizon from the raw text.
2. **Structured Context Node:** Executes live API calls to `yfinance` to fetch OHLCV data and calculates technical indicators (MACD, RSI, Moving Averages, Volatility).
3. **RAG Context Node:** Executes live API calls to `NewsData.io` to scrape real-time financial news, embeds the text using `mistral-embed`, and stores it in a **FAISS** vector database for context retrieval.
4. **LLM Reasoning Node:** Injects all multi-modal context (structured math + unstructured news) into **Mistral-Large**, which outputs a highly structured JSON portfolio recommendation.

---

## 🔬 Historical Backtesting ("Time Travel" Architecture)

To mathematically prove that unstructured news data (RAG) actually improves the AI's predictions, we built a sophisticated Quantitative Evaluation pipeline.

### The Ablation Study (Decision Delta)
The `Historical Analyst` UI allows you to run the Agent in a strict, sandboxed historical environment (e.g., locking its knowledge base to exactly March 31, 2026). 
The pipeline runs an automated A/B test:
- **Version A (Structured Only):** The LLM evaluates the stock using *only* math (MACD, RSI).
- **Version B (Structured + RAG):** The LLM evaluates the stock using math *plus* historical news.

### Ground Truth Validation
The system then compares the Agent's predicted stock direction against the **actual** stock market returns over the prediction horizon (e.g., +15% actual return vs Predicted "BUY"), establishing a definitive Win Rate for the Agent.

---

## 🚀 Setup & Launch Instructions

Follow these steps to run the Agentic architecture locally:

### 1. Extract and Navigate
Extract the repository and open your terminal/command prompt inside the folder:
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

### 4. Configure API Keys
The Agent requires LLM and News API access to function.
1. Copy the `.env.example` file and rename it to `.env`.
2. Open `.env` and add your API keys:
   ```env
   # Select your LLM Provider (options: mistral, grok, openai, gemini)
   LLM_PROVIDER=mistral

   # Enter the corresponding API Key
   MISTRAL_API_KEY=your_mistral_key
   
   # Required for Live News RAG
   NEWSDATA_API_KEY=your_newsdata_api_key
   ```

### 5. Launch the Application
```bash
streamlit run ui/app.py
```
Open `http://localhost:8501` in your browser.

---

## 💻 Using the Application

1. **Live Agent Tab:** Chat naturally with the AI. Ask it to analyze any of the 16 supported Nifty stocks (e.g., TCS, HDFC Bank, Sun Pharma, Maruti Suzuki), assign a budget, and watch it execute its LangGraph pipeline in real-time.
2. **Historical Backtesting:** Run ablation studies on past dates to mathematically verify the impact of the RAG pipeline against actual historical returns.
3. **Performance Dashboard:** View the aggregate portfolio metrics, RAG impact statistics, and overall win-rate for all backtests executed in your session.
