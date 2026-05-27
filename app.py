# =========================================================
# HOUSE PRICE PREDICTION PROJECT USING STREAMLIT
# Upload CSV / Excel File and Predict House Prices
# =========================================================

# STEP 1:
# Install Required Libraries
#
# pip install streamlit pandas scikit-learn openpyxl
#
# STEP 2:
# Save this file as app.py
#
# STEP 3:
# Run using:
# streamlit run app.py
# =========================================================

import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# =========================================================
# STREAMLIT TITLE
# =========================================================

st.title("🏠 House Price Prediction App")

st.write("Upload a CSV or Excel file containing house data.")

# =========================================================
# FILE UPLOAD
# =========================================================

uploaded_file = st.file_uploader(
    "Upload CSV or Excel File",
    type=["csv", "xlsx"]
)

# =========================================================
# PROCESS FILE
# =========================================================

if uploaded_file is not None:

    # Read file
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)

    else:
        df = pd.read_excel(uploaded_file)

    # =====================================================
    # SHOW DATA
    # =====================================================

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # =====================================================
    # COLUMN SELECTION
    # =====================================================

    st.subheader("Select Target Column")

    target_column = st.selectbox(
        "Choose House Price Column",
        df.columns
    )

    # =====================================================
    # PREPARE DATA
    # =====================================================

    X = df.drop(columns=[target_column])

    y = df[target_column]

    # =====================================================
    # HANDLE CATEGORICAL DATA
    # =====================================================

    X = pd.get_dummies(X)

    # =====================================================
    # TRAIN TEST SPLIT
    # =====================================================

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # =====================================================
    # TRAIN MODEL
    # =====================================================

    model = LinearRegression()

    model.fit(X_train, y_train)

    # =====================================================
    # PREDICTIONS
    # =====================================================

    predictions = model.predict(X_test)

    # =====================================================
    # MODEL EVALUATION
    # =====================================================

    mae = mean_absolute_error(y_test, predictions)

    r2 = r2_score(y_test, predictions)

    # =====================================================
    # SHOW RESULTS
    # =====================================================

    st.subheader("Model Performance")

    st.write(f"Mean Absolute Error : {mae:.2f}")

    st.write(f"R2 Score : {r2:.2f}")

    # =====================================================
    # PREDICTION TABLE
    # =====================================================

    result_df = pd.DataFrame({
        "Actual Price": y_test.values,
        "Predicted Price": predictions
    })

    st.subheader("Prediction Results")

    st.dataframe(result_df.head(20))

    # =====================================================
    # MANUAL INPUT PREDICTION
    # =====================================================

    st.subheader("Predict New House Price")

    input_data = {}

    for column in X.columns:

        input_data[column] = st.number_input(
            f"Enter {column}",
            value=0.0
        )

    # Convert input to DataFrame
    input_df = pd.DataFrame([input_data])

    # Prediction button
    if st.button("Predict Price"):

        predicted_price = model.predict(input_df)

        st.success(
            f"Predicted House Price: ₹ {predicted_price[0]:,.2f}"
        )

# =========================================================
# END OF PROJECT
# =========================================================