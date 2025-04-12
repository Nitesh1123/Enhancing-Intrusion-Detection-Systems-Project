# Intrusion Detection System (IDS) with Supervised Learning

A machine learning-based IDS to classify network traffic as normal or anomalous using the NSL-KDD dataset. Achieves 99.76% accuracy with a Random Forest Classifier and includes a Streamlit app for interactive predictions and visualizations.

## Features
- Data preprocessing with categorical encoding and feature scaling.
- Exploratory Data Analysis (EDA) with pie charts, heatmaps, and scatter plots.
- Random Forest model with 99.76% validation accuracy.
- Streamlit web app for uploading data, predicting intrusions, and viewing visualizations (feature importance, correlation heatmaps, boxplots).

## Dataset
- **NSL-KDD**: Sourced from [Kaggle](https://www.kaggle.com/datasets/sampadab17/network-intrusion-detection).
- Training: 25,192 rows, 41 features, binary target (normal/anomaly).

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/IDS-Supervised-Learning.git