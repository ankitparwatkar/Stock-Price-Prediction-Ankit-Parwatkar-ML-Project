import streamlit as st
import numpy as np
import pandas as pd
import pickle

# Load dataset
df = pd.read_csv("ADANIPORTS.csv")
x = df.iloc[:, 3:8].values
y = df.iloc[:, 8].values

# Split dataset
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

# Train model
from sklearn.svm import SVR
model = SVR(kernel='poly', degree=2, C=2)
model.fit(x_train, y_train)

# Save the model
with open('stock.pkl', 'wb') as file:
    pickle.dump(model, file)

# Load model
with open('stock.pkl', 'rb') as file:
    model = pickle.load(file)

# Streamlit UI
st.set_page_config(page_title="Stock Price Prediction", layout="wide")
st.title("📈 Stock Price Prediction App")
st.markdown("### Predict future stock prices based on past performance!")

# Sidebar Inputs
st.sidebar.header("Enter Stock Data:")
Prev_Close = st.sidebar.number_input("Prev Close", value=440.0)
Open = st.sidebar.number_input("Open", value=770.0)
High = st.sidebar.number_input("High", value=1050.0)
Low = st.sidebar.number_input("Low", value=770.0)
Last = st.sidebar.number_input("Last", value=959.0)

# Prediction
if st.sidebar.button("🔮 Predict"):
    result = model.predict([[Prev_Close, Open, High, Low, Last]])[0]
    
    st.success(f"### Predicted Stock Price: **{result:.2f}** 💰")

    # Performance Metrics
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    y_pred = model.predict(x_test)
    MAE = mean_absolute_error(y_test, y_pred)
    MSE = mean_squared_error(y_test, y_pred)
    RMSE = np.sqrt(MSE)

    # Display Metrics using Columns
    col1, col2, col3 = st.columns(3)
    col1.metric(label="📉 Mean Absolute Error", value=f"{MAE:.2f}")
    col2.metric(label="📊 Mean Squared Error", value=f"{MSE:.2f}")
    col3.metric(label="📈 Root Mean Squared Error", value=f"{RMSE:.2f}")

    # Plot Feature Importance (Dummy Example)
    st.markdown("### Feature Contribution")
    st.bar_chart(pd.DataFrame({"Feature": ["Prev Close", "Open", "High", "Low", "Last"], "Impact": [0.2, 0.3, 0.25, 0.15, 0.1]}))

# Footer
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("🚀 Developed with ❤️ using **Streamlit & Scikit-Learn**")
