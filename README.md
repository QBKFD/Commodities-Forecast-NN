#  Commodities Forecasting with Neural Networks

This repository contains the full implementation of my Master's thesis:  
**"Forecasting Commodity Markets Using Neural Networks: A Quantitative Study Using Engineered Inputs and Deep Learning Architectures."**

The project explores the use of deep learning architectures—including LSTM, GRU, Transformer-based models, and hybrid variants—for **1-day ahead forecasting** of major commodities: **Gold**, **Silver**, and **Copper**.


##  Key Contributions

-  **Custom Feature Engineering**: Created from raw market data and enriched with macroeconomic and geopolitical indicators
-  **Dual-Input Model Design**: Combines sequential price data with static macro/sentiment features across multiple architectures
-  Evaluates a **wide variety of architectures**, including Transformer-based models (Autoformer, Informer, PatchTST, TimesNet), recurrent networks (LSTM, GRU), and hybrid variants (e.g., TFT+GRU)
-  Walk-forward validation for robust out-of-sample performance
-  Comprehensive evaluation using RMSE, MAE, and Directional Accuracy
