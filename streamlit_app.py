import streamlit as st
import pandas as pd
import joblib
import os

# ---------------------------------------
# 1. PAGE CONFIGURATION
# ---------------------------------------
st.set_page_config(
    page_title="Health Insurance Predictor",
    page_icon="🏥",
    layout="wide"
)

# ---------------------------------------
# 2. LOAD MODEL
# ---------------------------------------
@st.cache_resource
def load_model():
    model = joblib.load("models/best_model.pkl")
    feature_columns = joblib.load(
        "models/feature_columns.pkl"
    )
    return model, feature_columns
model, feature_columns = load_model()

# ---------------------------------------
# 3. APPLICATION HEADER
# ---------------------------------------
st.title("🏥 Health Insurance Cost Prediction")
st.write(
    "Predict estimated health insurance charges "
    "using Machine Learning."
)
st.markdown("---")

# ---------------------------------------
# 4. USER INPUT SECTION
# ---------------------------------------
st.subheader("Enter Your Personal Details")
col1, col2 = st.columns(2)
with col1:
    age = st.number_input(
        "Age",
        min_value=18,
        max_value=64,
        value=25
    )
    sex = st.selectbox(
        "Gender",
        ["male", "female"]
    )
    bmi = st.number_input(
        "BMI",
        min_value=10.0,
        max_value=60.0,
        value=25.0
    )
with col2:
    children = st.number_input(
        "Number of Children",
        min_value=0,
        max_value=5,
        value=0
    )
    smoker = st.selectbox(
        "Smoking Status",
        ["no", "yes"]
    )
    region = st.selectbox(
        "Region",
        [
            "southwest",
            "southeast",
            "northwest",
            "northeast"
        ]
    )

# ---------------------------------------
# 5. PREDICTION BUTTON
# ---------------------------------------
st.markdown("---")
if st.button("Predict Insurance Cost"):

    if age < 18 or age > 64:
        st.error("Age must be between 18 and 64.")

    elif bmi <= 0:
        st.error("Please enter a valid BMI.")

    else:

        user_data = pd.DataFrame({

            "age": [age],
            "sex": [sex],
            "bmi": [bmi],
            "children": [children],
            "smoker": [smoker],
            "region": [region]

        })

        user_encoded = pd.get_dummies(user_data)

        user_encoded = user_encoded.reindex(
            columns=feature_columns,
            fill_value=0
        )

        prediction = model.predict(user_encoded)[0]

        st.success("Prediction Generated Successfully!")

        st.metric(
            "Estimated Insurance Cost",
            f"${prediction:,.2f}"
        )
    # Generate prediction
    prediction = model.predict(user_encoded)[0]

    # -----------------------------------
    # 6. DISPLAY RESULT
    # -----------------------------------
    st.success("Prediction Generated Successfully!")
    st.subheader("Estimated Insurance Charges")
    st.metric(
        label="Predicted Insurance Cost",
        value=f"${prediction:,.2f}"
    )
    st.info(
        "This is an estimated prediction based on "
        "your trained machine learning model, "
        "not an actual insurance quotation."
    )

# ---------------------------------------
# 7. FOOTER
# ---------------------------------------
st.markdown("---")
st.caption(
    "Health Insurance Cost Prediction System | "
    "Powered by Gradient Boosting Machine Learning"
)