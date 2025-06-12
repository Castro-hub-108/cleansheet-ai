import streamlit as st
from cleaner import clean_data
from profiler import generate_report

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
