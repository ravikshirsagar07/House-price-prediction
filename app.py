# =========================================================
# HOUSE PRICE PREDICTION STREAMLIT APP
# Supports CSV, XLSX, XLS
# =========================================================

# =========================================================
# IMPORT LIBRARIES
# =========================================================

import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="House Price Prediction",
    layout="wide"
)

# =========================================================
# TITLE
# =========================================================

st.title("🏠 House Price Prediction App")

st.write(
    "Upload CSV, XLSX, or XLS dataset file"
)

# =========================================================
# FILE UPLOADER
# =========================================================

uploaded_file = st.file_uploader(
    "Upload Dataset",
    type=["csv", "xlsx", "xls"]
)

# =========================================================
# PROCESS FILE
# =========================================================

if uploaded_file is not None:

    try:

        # =================================================
        # FILE NAME
        # =================================================

        file_name = uploaded_file.name.lower()

        # =================================================
        # READ CSV FILE
        # =================================================

        if file_name.endswith(".csv"):

            df = pd.read_csv(uploaded_file)

        # =================================================
        # READ XLSX FILE
        # =================================================

        elif file_name.endswith(".xlsx"):

            df = pd.read_excel(
                uploaded_file,
                engine="openpyxl"
            )

        # =================================================
        # READ XLS FILE
        # =================================================

        elif file_name.endswith(".xls"):

            try:

                # Try real XLS file
                df = pd.read_excel(
                    uploaded_file,
                    engine="xlrd"
                )

            except:

                # Some fake XLS files are actually CSV
                uploaded_file.seek(0)

                df = pd.read_csv(uploaded_file)

        # =================================================
        # SUCCESS MESSAGE
        # =================================================

        st.success("File uploaded successfully!")

        # =================================================
        # SHOW DATASET
        # =================================================

        st.subheader("Dataset Preview")

        st.dataframe(df.head())

        # =================================================
        # DATASET INFO
        # =================================================

        st.subheader("Dataset Information")

        st.write("Rows :", df.shape[0])

        st.write("Columns :", df.shape[1])

        st.write("Column Names :")

        st.write(list(df.columns))

        # =================================================
        # TARGET COLUMN
        # =================================================

        st.subheader("Select Target Column")

        target_column = st.selectbox(
            "Choose House Price Column",
            df.columns
        )

        # =================================================
        # FEATURES AND TARGET
        # =================================================

        X = df.drop(columns=[target_column])

        y = df[target_column]

        # =================================================
        # HANDLE CATEGORICAL DATA
        # =================================================

        X = pd.get_dummies(X)

        # =================================================
        # TRAIN TEST SPLIT
        # =================================================

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42
        )

        # =================================================
        # MODEL TRAINING
        # =================================================

        model = LinearRegression()

        model.fit(X_train, y_train)

        # =================================================
        # PREDICTIONS
        # =================================================

        predictions = model.predict(X_test)

        # =================================================
        # MODEL PERFORMANCE
        # =================================================

        mae = mean_absolute_error(
            y_test,
            predictions
        )

        r2 = r2_score(
            y_test,
            predictions
        )

        # =================================================
        # SHOW PERFORMANCE
        # =================================================

        st.subheader("Model Performance")

        st.write(
            f"Mean Absolute Error : {mae:.2f}"
        )

        st.write(
            f"R2 Score : {r2:.2f}"
        )

        # =================================================
        # RESULTS TABLE
        # =================================================

        results_df = pd.DataFrame({

            "Actual Price": y_test.values,

            "Predicted Price": predictions

        })

        st.subheader("Prediction Results")

        st.dataframe(results_df.head(20))

        # =================================================
        # MANUAL PREDICTION
        # =================================================

        st.subheader("Predict New House Price")

        input_data = {}

        for column in X.columns:

            input_data[column] = st.number_input(
                f"Enter {column}",
                value=0.0
            )

        input_df = pd.DataFrame([input_data])

        # =================================================
        # PREDICT BUTTON
        # =================================================

        if st.button("Predict House Price"):

            predicted_price = model.predict(input_df)

            st.success(
                f"Predicted House Price : ₹ {predicted_price[0]:,.2f}"
            )

    # =====================================================
    # ERROR HANDLING
    # =====================================================

    except Exception as e:

        st.error("Error while processing file")

        st.write(e)

# =========================================================
# NO FILE MESSAGE
# =========================================================

else:

    st.info(
        "Please upload CSV, XLSX, or XLS file"
    )

# =========================================================
# END
# =========================================================
