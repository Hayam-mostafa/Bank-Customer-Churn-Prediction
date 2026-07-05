# Bank Customer Churn Prediction

## Overview
This project predicts customer churn for a bank using machine learning.
The goal is to identify customers likely to leave and support retention strategies.

## Dataset
The dataset includes customer demographics, banking behavior, and engagement features.
Target: `Exited` (1 = churn, 0 = stay)

## Workflow
1. Data cleaning & preprocessing
2. Exploratory data analysis (EDA)
3. Feature encoding
4. Train/test split
5. Model training (LightGBM)
6. Evaluation

## Model
**LightGBM Classifier** — chosen for the best balance of recall, F1-score, and ROC-AUC.

## Results
- ROC-AUC: ~0.87
- Strong recall for churn detection
- Good overall balance between metrics

## Key insights
- Older customers are more likely to churn
- Higher balance increases churn probability
- Active members are less likely to leave
- Complaints strongly indicate churn (excluded from the model to avoid leakage)

## Interactive demo
This project includes a Gradio web app (`app.py`) for live predictions.

![App screenshot](Test.png)

### Running it locally
\```bash
pip install -r requirements.txt
python app.py
\```

## Tools
Python, Pandas, Scikit-learn, LightGBM, Seaborn, Matplotlib, Gradio

## Business impact
Helps the bank identify at-risk customers and improve retention strategies.
