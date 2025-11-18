import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from dotenv import load_dotenv
import google.generativeai as genai
import os

# FIRST Streamlit command
st.set_page_config(page_title="Smart Budget Assistant", layout="wide")

# Load env
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Load ML model
@st.cache_resource
def load_model():
    return joblib.load("budget_category_model.joblib")

model_bundle = load_model()
vectorizer = model_bundle["vectorizer"]
clf = model_bundle["model"]
TEXT_COL = model_bundle["text_col"]

# LLM domain prompt
DOMAIN_PROMPT = """
You are a financial advisor...

Your job is to help users understand their expenses
and allocate their budget across essentials, lifestyle,
rent, transport, utilities, savings, investments, etc.

Use the uploaded CSV summary to give personalized advice.
"""

st.title("📊 Smart Budget Assistant")
st.subheader("Upload your transaction CSV")

uploaded_file = st.file_uploader("Choose a CSV", type="csv")

expense_summary_text = ""

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # Clean text column
    if TEXT_COL not in df.columns:
        st.error(f"Column '{TEXT_COL}' not found. Your CSV must contain transaction text.")
    else:
        df[TEXT_COL] = df[TEXT_COL].astype(str).str.lower()

        # Vectorize + predict
        X_user = vectorizer.transform(df[TEXT_COL])
        preds = clf.predict(X_user)

        df["broad_category"] = preds

        st.subheader("Tagged Transactions")
        st.dataframe(df)

        # Expense chart
        if "amount" in df.columns:
            df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0)
            by_cat = df.groupby("broad_category")["amount"].sum()

            fig, ax = plt.subplots()
            by_cat.plot(kind="pie", autopct="%1.1f%%", ax=ax)
            ax.set_ylabel("")
            st.pyplot(fig)

            expense_summary_text = f"\nSpending by category:\n{by_cat.to_dict()}"

st.subheader("Ask your Budget Assistant")

user_input = st.chat_input("Ask something about your finances...")

if user_input:
    st.chat_message("user").markdown(user_input)

    prompt = f"""
    {DOMAIN_PROMPT}

    User query:
    {user_input}

    Uploaded expense summary:
    {expense_summary_text}
    """

    try:
        llm = genai.GenerativeModel("gemini-2.0-flash")
        response = llm.generate_content(prompt)
        reply = response.text
    except Exception as e:
        reply = f"Gemini error: {e}"

    st.chat_message("assistant").markdown(reply)
