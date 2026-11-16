import streamlit as st

st.set_page_config(
    page_title="Dashboard Test",
    layout="wide"
)

st.title("🏥 Health Insurance Analytics Dashboard")

st.success("Streamlit is working successfully!")

st.header("Dashboard Testing")

col1, col2, col3 = st.columns(3)

col1.metric("Total Customers", "1338")
col2.metric("Average Age", "39")
col3.metric("Average BMI", "30.6")

st.subheader("Testing Chart")

import pandas as pd

df = pd.DataFrame({
    "Age": [20, 25, 30, 35, 40],
    "Charges": [2000, 3000, 4500, 6000, 8000]
})

st.line_chart(
    df,
    x="Age",
    y="Charges"
)

st.write("Dashboard is running!")