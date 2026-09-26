import pandas as pd
import streamlit as st
import tensorflow as tf
import pickle

# Page Configuration
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="centered"
)

# Load Preprocessing Objects and Model
@st.cache_resource
def load_artifacts():

    with open("onehot_encoder_geo.pkl", "rb") as file:
        ohe_geo = pickle.load(file)

    with open("label_encoder_gender.pkl", "rb") as file:
        le_gender = pickle.load(file)

    with open("scaler.pkl", "rb") as file:
        scaler = pickle.load(file)

    model = tf.keras.models.load_model("model.h5")

    return ohe_geo, le_gender, scaler, model

ohe_geo, le_gender, scaler, model = load_artifacts()

# Streamlit App
st.title("📊 Customer Churn Prediction")

st.write(
    "Enter the customer's information below to predict "
    "the probability that the customer will churn."
)

# User Input Form
with st.form("customer_form"):
    st.subheader("Customer Information")
    col1, col2 = st.columns(2)
    with col1:
        geography = st.selectbox(
            "Geography",
            ohe_geo.categories_[0]
        )
        gender = st.selectbox(
            "Gender",
            le_gender.classes_
        )
        age = st.slider(
            "Age",
            min_value=18,
            max_value=92,
            value=35
        )
        tenure = st.slider(
            "Tenure",
            min_value=0,
            max_value=10,
            value=5
        )
        num_of_products = st.slider(
            "Number of Products",
            min_value=1,
            max_value=4,
            value=1
        )

    with col2:
        credit_score = st.number_input(
            "Credit Score",
            min_value=300,
            max_value=850,
            value=650
        )
        balance = st.number_input(
            "Balance",
            min_value=0.0,
            value=50000.0,
            step=1000.0
        )
        estimated_salary = st.number_input(
            "Estimated Salary",
            min_value=0.0,
            value=50000.0,
            step=1000.0
        )
        has_cr_card = st.selectbox(
            "Has Credit Card?",
            [0, 1],
            format_func=lambda x: "Yes" if x == 1 else "No"
        )
        is_active_member = st.selectbox(
            "Is Active Member?",
            [0, 1],
            format_func=lambda x: "Yes" if x == 1 else "No"
        )

    # Prediction Button
    submitted = st.form_submit_button(
        "🔮 Predict Churn",
        use_container_width=True
    )

# Prediction
if submitted:
    # Encode Gender
    gender_encoded = le_gender.transform([gender])[0]

    # Create Input DataFrame
    input_data = pd.DataFrame(
        {
            "CreditScore": [credit_score],
            "Gender": [gender_encoded],
            "Age": [age],
            "Tenure": [tenure],
            "Balance": [balance],
            "NumOfProducts": [num_of_products],
            "HasCrCard": [has_cr_card],
            "IsActiveMember": [is_active_member],
            "EstimatedSalary": [estimated_salary]
        }
    )
    
    # One-Hot Encode Geography
    geo_encoded = ohe_geo.transform([[geography]])
    geo_encoded_df = pd.DataFrame(
        geo_encoded,
        columns=ohe_geo.get_feature_names_out(["Geography"])
    )

    # Combine Features
    input_data = pd.concat(
        [
            input_data.reset_index(drop=True),
            geo_encoded_df
        ],
        axis=1
    )

    # Scale Input
    input_data_scaled = scaler.transform(input_data)

    # Make Prediction
    prediction = model.predict(
        input_data_scaled,
        verbose=0
    )
    prediction_probability = float(prediction[0][0])

    # Display Result
    st.subheader("Prediction Result")
    probability_percentage = prediction_probability * 100
    st.metric(
        "Churn Probability",
        f"{probability_percentage:.2f}%"
    )
    st.progress(prediction_probability)

    # Decision
    if prediction_probability >= 0.5:
        st.error(
            "⚠️ The customer is likely to churn."
        )
    else:
        st.success(
            "✅ The customer is not likely to churn."
        )