import os
import pandas as pd
import streamlit as st


LOG_FILE_PATH = "monitoring/prediction_logs.csv"


st.set_page_config(
    page_title="Customer Churn Monitoring Dashboard",
    layout="wide"
)

st.title("Customer Churn Monitoring Dashboard")

if not os.path.exists(LOG_FILE_PATH):
    st.warning("No prediction logs found yet.")
    st.stop()

df = pd.read_csv(LOG_FILE_PATH)

st.subheader("Prediction Summary")

col1, col2, col3 = st.columns(3)

total_predictions = len(df)
churn_count = (df["prediction"] == "Churn").sum()
no_churn_count = (df["prediction"] == "No Churn").sum()
avg_probability = df["churn_probability"].mean()

col1.metric("Total Predictions", total_predictions)
col2.metric("Churn Predictions", churn_count)
col3.metric("No Churn Predictions", no_churn_count)

st.metric("Average Churn Probability", round(avg_probability, 4))

st.subheader("Churn vs No Churn Distribution")
prediction_counts = df["prediction"].value_counts()
st.bar_chart(prediction_counts)

st.subheader("Recent Prediction Logs")
st.dataframe(df.tail(10), use_container_width=True)

st.subheader("Raw Logs")
st.dataframe(df, use_container_width=True)