import gradio as gr
import pandas as pd
import joblib

try:
    model = joblib.load("model/churn_model.pkl")
    model_loaded = True
except Exception:
    model = None
    model_loaded = False


def build_input_row(credit_score, geography, gender, age, tenure, balance,
                     num_products, has_cr_card, is_active, estimated_salary,
                     satisfaction, card_type, points_earned):
    row = {
        "CreditScore": credit_score,
        "Geography": geography,
        "Gender": gender,
        "Age": age,
        "Tenure": tenure,
        "Balance": balance,
        "NumOfProducts": num_products,
        "HasCrCard": 1 if has_cr_card == "Yes" else 0,
        "IsActiveMember": 1 if is_active == "Yes" else 0,
        "EstimatedSalary": estimated_salary,
        "Satisfaction Score": satisfaction,
        "Card Type": card_type,
        "Point Earned": points_earned,
    }
    return pd.DataFrame([row])


def preprocess(raw_df):
    df = raw_df.copy()
    df["Gender"] = df["Gender"].map({"Female": 0, "Male": 1})
    df = pd.get_dummies(df, columns=["Geography", "Card Type"])

    if model_loaded and hasattr(model, "feature_name_"):
        df = df.reindex(columns=list(model.feature_name_), fill_value=0)

    return df


def predict_churn(credit_score, geography, gender, age, tenure, balance,
                   num_products, has_cr_card, is_active, estimated_salary,
                   satisfaction, card_type, points_earned):

    input_df = build_input_row(
        credit_score, geography, gender, age, tenure, balance,
        num_products, has_cr_card, is_active, estimated_salary,
        satisfaction, card_type, points_earned
    )
    processed_df = preprocess(input_df)

    try:
        proba = model.predict_proba(processed_df)[0][1]
        pct = round(proba * 100, 1)
        is_high_risk = proba >= 0.5
        pill_text = "Likely to leave" if is_high_risk else "Likely to stay"

        return f"""
        <div class="result-card">
            <div class="result-number">{pct}%</div>
            <span class="result-pill">{pill_text}</span>
        </div>
        """

    except Exception as e:
        return f"""<div class="result-card">Prediction failed: {e}</div>"""

custom_css = """
#hero-title {
    font-weight: 700;
    font-size: 34px;
    text-align: center;
}

#hero-sub {
    font-size: 15px;
    text-align: center;
    margin: 4px 0 20px;
}

.gradio-container label,
.gradio-container label span,
.gradio-container .label-wrap,
.gradio-container .label-wrap span,
.gradio-container span[data-testid="block-info"] {
    font-weight: 700 !important;
    color: #000000 !important;
}

button[aria-label*="Reset"],
button[aria-label*="reset"],
button[aria-label*="Refresh"],
button[aria-label*="refresh"],
.icon-button {
    display: none !important;
}

.result-card {
    border-radius: 18px;
    padding: 30px 32px;
    text-align: center;
}

.result-number {
    font-weight: 700;
    font-size: 54px;
    margin: 4px 0;
}

.result-pill {
    font-weight: 600;
    padding: 6px 16px;
    border-radius: 999px;
    display: inline-block;
}
"""

theme = gr.themes.Default()

with gr.Blocks(title="Churn Prediction") as demo:

    gr.HTML("""
        <div id="hero-title">Will this customer stay?</div>
        <div id="hero-sub">Fill in the details below and find out in one click.</div>
    """)

    with gr.Row():
        with gr.Column():
            credit_score = gr.Slider(300, 900, value=650, step=1, label="Credit Score")
            age = gr.Slider(18, 100, value=35, step=1, label="Age")
            tenure = gr.Slider(0, 10, value=3, step=1, label="Tenure, years")
            balance = gr.Number(value=50000, label="Balance")
            satisfaction = gr.Slider(1, 5, value=3, step=1, label="Satisfaction Score")
            points_earned = gr.Slider(0, 1000, value=500, step=1, label="Points Earned")

        with gr.Column():
            geography = gr.Dropdown(["France", "Germany", "Spain"], value="France", label="Geography")
            gender = gr.Dropdown(["Female", "Male"], value="Female", label="Gender")
            num_products = gr.Slider(1, 4, value=1, step=1, label="Number of Products")
            has_cr_card = gr.Dropdown(["Yes", "No"], value="Yes", label="Has Credit Card")
            is_active = gr.Dropdown(["Yes", "No"], value="Yes", label="Active Member")
            card_type = gr.Dropdown(["DIAMOND", "GOLD", "PLATINUM", "SILVER"], value="GOLD", label="Card Type")

    estimated_salary = gr.Number(value=60000, label="Estimated Salary")

    predict_btn = gr.Button("Predict", variant="primary")
    result_html = gr.HTML()

    predict_btn.click(
        fn=predict_churn,
        inputs=[
            credit_score, geography, gender, age, tenure, balance,
            num_products, has_cr_card, is_active, estimated_salary,
            satisfaction, card_type, points_earned
        ],
        outputs=result_html
    )


if __name__ == "__main__":
    demo.launch(theme=theme, css=custom_css)
