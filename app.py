from cleaner import clean_data
from profiler import generate_report
import streamlit as st
import pandas as pd
import plotly.express as px
import io
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Dark Mode Toggle
theme = st.sidebar.selectbox("🎨 Select Theme", ["Light", "Dark"])

# Theme Styles
if theme == "Dark":
    st.markdown(
        """
        <style>
        body { background-color: #1e1e1e; color: white; }
        .stApp { background-color: #1e1e1e; color: white; }
        .stDataFrame, .stTable, .css-1d391kg { background-color: #1e1e1e !important; color: white !important; }
        </style>
        """,
        unsafe_allow_html=True
    )

st.title("CleanSheet AI - Your Data Analysis Assistant")

uploaded_file = st.file_uploader("Upload your CSV or Excel file", type=['csv', 'xlsx'])

if uploaded_file:
    # Read CSV or Excel file
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.write("### Data Preview")

    st.dataframe(df.head())

    st.write("### 📊 Data Visualization")

    st.write("### 📈 Summary Statistics & 📥 Download")

    st.write("### 🤖 Simple Machine Learning: Linear Regression")

    if uploaded_file:
        num_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()

        if len(num_cols) >= 2:
            target = st.selectbox("Select the target column (what to predict)", options=num_cols)
            features = st.multiselect("Select feature columns (inputs)",
                                      options=[col for col in num_cols if col != target])

            if features and target:
                X = df[features]
                y = df[target]

                # Train/test split
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                # Train model
                model = LinearRegression()
                model.fit(X_train, y_train)

                # Predict
                predictions = model.predict(X_test)

                # Show predictions
                results_df = X_test.copy()
                results_df["Actual"] = y_test
                results_df["Predicted"] = predictions
                st.write("### 📊 Prediction Results")
                st.dataframe(results_df)

                # Show evaluation
                mse = mean_squared_error(y_test, predictions)
                st.write(f"📉 Mean Squared Error: `{mse:.2f}`")

                # Download predictions
                pred_csv = results_df.to_csv().encode('utf-8')
                st.download_button("📥 Download Predictions", pred_csv, "predictions.csv", "text/csv")

    if uploaded_file:
        # Get numeric columns
        num_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()

        if num_cols:
            # Multiselect for columns
            selected_cols = st.multiselect("Select numeric columns to summarize", options=num_cols, default=num_cols)

            if selected_cols:
                # Compute summary stats
                summary_df = df[selected_cols].describe().T
                summary_df["median"] = df[selected_cols].median()

                # Display table
                st.dataframe(summary_df)

                # Create and enable CSV download
                csv = summary_df.to_csv().encode('utf-8')
                st.download_button(
                    label="📥 Download Summary as CSV",
                    data=csv,
                    file_name='summary_statistics.csv',
                    mime='text/csv',
                )
            else:
                st.info("👉 Please select at least one numeric column.")
        else:
            st.warning("⚠️ No numeric columns found in this dataset.")

    if uploaded_file:
        chart_type = st.selectbox("Choose a chart type", ["Bar", "Line", "Scatter", "Pie"])
        numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
        all_columns = df.columns.tolist()

        if chart_type in ["Bar", "Line", "Scatter"]:
            x_col = st.selectbox("X-Axis", options=all_columns)
            y_col = st.selectbox("Y-Axis", options=numeric_columns)

            if chart_type == "Bar":
                fig = px.bar(df, x=x_col, y=y_col)
            elif chart_type == "Line":
                fig = px.line(df, x=x_col, y=y_col)
            else:
                fig = px.scatter(df, x=x_col, y=y_col)

            st.plotly_chart(fig, use_container_width=True)

        elif chart_type == "Pie":
            label_col = st.selectbox("Labels (categorical)", options=all_columns)
            value_col = st.selectbox("Values (numeric)", options=numeric_columns)
            fig = px.pie(df, names=label_col, values=value_col)
            st.plotly_chart(fig, use_container_width=True)

    # Now you can add more analysis below using `df`
else:
    st.info("Please upload a CSV or Excel file to start analysis.")

st.title("🧼 CleanSheet AI")
st.subheader("Smarter, faster, cleaner data")

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file:
    st.success("File uploaded successfully.")
    df_cleaned, log = clean_data(uploaded_file)
    st.write("✅ Cleaning Summary:")
    st.json(log)

    report_figs = generate_report(df_cleaned)
    for fig in report_figs:
        st.pyplot(fig)

    st.download_button("Download Cleaned CSV", df_cleaned.to_csv(index=False), "cleaned_data.csv")
