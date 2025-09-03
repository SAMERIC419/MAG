import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="LTFU Test", layout="wide")
st.title("📊 LTFU Test App")

st.write("This is a test app to check if Streamlit is working properly.")

# Simple test
if st.button("Test Button"):
    st.success("✅ Streamlit is working!")

# Test data loading
st.subheader("Test Data")
test_data = pd.DataFrame({
    'A': [1, 2, 3, 4, 5],
    'B': [10, 20, 30, 40, 50]
})
st.dataframe(test_data)

st.write("If you can see this, the basic app is working!")
