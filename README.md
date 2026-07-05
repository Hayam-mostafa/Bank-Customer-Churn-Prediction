# Bank Customer Churn Prediction
A machine learning project that predicts whether a bank customer is likely to leave the bank (churn) using customer information and banking behavior.
## Features
- Data preprocessing and cleaning
- Exploratory Data Analysis (EDA)
- Feature engineering
- LightGBM classification model
- Model evaluation
- Interactive Gradio web application
## Dataset
The dataset contains customer demographics, account information, and banking activity.
**Target:**
- `Exited = 1` → Customer churned
- `Exited = 0` → Customer stayed
## Model
**LightGBM Classifier** was selected directly for this task due to:
- Strong native handling of categorical and tabular data
- Fast training speed on medium-sized datasets like this one
- Built-in support for feature importance analysis
- Proven track record for classification tasks with imbalanced targets (churn is typically imbalanced)
## Results
- ROC-AUC: **0.87**
- High recall for detecting churned customers
- Balanced overall performance
## Key Insights
- Older customers are more likely to churn
- Higher account balance increases churn probability
- Active members are less likely to leave
- Customer complaints strongly correlate with churn (excluded from model features to avoid data leakage)

##  Run the Project
```bash
pip install -r requirements.txt
python app.py
```
## Technologies
- Python
- Pandas
- Scikit-learn
- LightGBM
- Matplotlib
- Seaborn
- Gradio
## Business Value
This project helps banks identify customers who are at risk of leaving, allowing them to improve customer retention strategies.
