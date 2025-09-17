# VIXNLP: Volatility Index Prediction with Natural Language Processing

## 📌 Status

This was my first major exploration project in quantitative finance and natural language processing. The work has been discontinued, and I have since moved on to more advanced projects that are no longer developed publicly on GitHub. I’m leaving this repository as a record of my initial experimentation with VIX forecasting, sentiment features, and regime analysis.

---

## 🔍 About This Project

The goal of this project was to investigate whether macroeconomic news sentiment could help forecast short-term movements and spikes in the CBOE Volatility Index (VIX).

The project combined **NLP-driven sentiment analysis** with **time-series modeling and classifiers** to study volatility regimes, spike events, and persistence of volatility levels.

While the pipeline was fully functional, it remained exploratory and was not productionized.

---

## 🛠️ Features

* **Daily sentiment ingestion & feature engineering**
* **Regime-labeled time series construction**
* **Markov switching models for regime detection**
* **Classifiers for regime/spike prediction**
* **Feedforward neural networks** for peak level prediction (no LSTMs used)
* **Spike detection and half-life analysis**

---

## 📊 Project Overview

1. **News Data Collection**

   * *MediaStack API* for financial news headlines
   * *Finnhub API* for company- and macro-specific news
   * A 7-year historical macro news dataset for backtesting

2. **Sentiment Analysis**

   * *FinBERT* for financial sentiment scoring
   * Headline vs. summary sentiment comparison
   * Custom calibration for financial domain context

3. **Feature Engineering**

   * Sentiment decay (rolling effects from prior news)
   * Negative shock intensity measures
   * Cross-features between sentiment and VIX levels

4. **Volatility Forecasting Models**

   * Markov regime-switching models (Normal vs. Panic states)
   * Logistic classifiers for regime transition prediction
   * Feedforward neural networks for spike peak prediction

5. **Spike Analysis**

   * Detection of volatility spikes
   * Duration and half-life estimation
   * Neural network-based peak level estimation

---

## 📈 Results

Key exploratory findings:

* Two distinct volatility regimes with differing baselines
* Negative news shocks had a strong influence on regime transitions
* Prior-day sentiment carried predictive power for next-day VIX changes
* Classifiers reached up to **\~94% accuracy** in distinguishing regimes
* Neural networks showed early promise in predicting spike peak levels, but were not used due to low data available and high probability of overfit

---

## 📂 Project Structure

* **/data/** → Processed datasets (excluded from repo due to size)
* **/finnhub\_news/** → Finnhub API integration
* **/utils/** → Helper functions
* **downloader.ipynb** → End-to-end pipeline notebook
* **mediastack\_news.py** → MediaStack API script
* **clean\_csv.py** → Data cleaning utilities

---

## ⚙️ Usage Notes

### Data

Large datasets (CSV, pickle) are excluded via `.gitignore`. You’ll need to regenerate them locally using the provided API scripts.

### Setup

Instructions for environment and dependencies are in `README_SETUP.md`.

### Running the Pipeline

Main workflow is available in `downloader.ipynb`:

1. Collect news data
2. Run sentiment analysis
3. Generate features
4. Train classifiers / neural networks
5. Run regime detection and spike analysis

---

## 🚀 Reflection

This project marked my entry into combining **NLP + financial time series modeling**. While I no longer actively maintain it, the lessons here shaped my later work in volatility forecasting, regime modeling, and model deployment. 
