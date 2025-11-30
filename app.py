import streamlit as st
import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split, cross_val_score

# Load dataset and model (initialize if not found)
try:
    data = joblib.load('data.joblib')
except:
    data = pd.DataFrame(columns=['Brand', 'Model', 'VehicleClass', 'Region', 'DriveTrain',
                                 'DealerCost', 'EngineSize', 'Cylinders', 'HorsePower',
                                 'Weight', 'Wheelbase', 'Length', 'MPG_Avg', 'MSRP'])

# Define preprocessing pipeline (same as in model.ipynb)
ordinal_col = ['VehicleClass']
ordinal_order = [['Sedan', 'SUV', 'Hybrid', 'Sports', 'Wagon', 'Truck']]
onehot_cols = ['Brand', 'Model', 'Region', 'DriveTrain']
impute_col = ['Cylinders']

trf1 = ColumnTransformer([
    ('impute_cy', SimpleImputer(strategy='mean'), impute_col),
    ('Veh_cla_oe', OrdinalEncoder(categories=ordinal_order), ordinal_col),
    ('ohe', OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore'), onehot_cols)
], remainder='passthrough')

def train_model(data):
    """Train and return a Ridge model pipeline"""
    x = data.drop(columns=['MSRP'])
    y = data['MSRP']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    model = Ridge(alpha=0.1)
    pipe = Pipeline([
        ('preprocess', trf1),
        ('model', model)
    ])
    pipe.fit(x_train, y_train)
    return pipe

# Train model on initial data
model = train_model(data)

# Streamlit UI
st.set_page_config(page_title="Online Car Price Learner", page_icon="🚗", layout="wide")
st.title("🚗 Car Price Analysis & Online Learning")

with st.sidebar:
    st.markdown("## 🧠 Online Learning Mode")
    st.image("https://img.icons8.com/fluency/96/car.png", width=100)
    page = st.radio("Choose a Feature", ["🏠 Predict MSRP", "📊 EDA Insights"])

if page == "🏠 Predict MSRP":
    st.header("📥 Submit Data for Prediction + Learning")

    col1, col2 = st.columns(2)

    with col1:
        brand = st.text_input("Brand")
        model_name = st.text_input("Model")
        vehicle_class = st.selectbox("Vehicle Class", ['Sedan', 'SUV', 'Hybrid', 'Sports', 'Wagon', 'Truck'])
        region = st.text_input("Region")
        drivetrain = st.text_input("DriveTrain")

    with col2:
        dealer_cost = st.number_input("Dealer Cost ($)", step=500.0)
        engine_size = st.number_input("Engine Size (L)", step=0.1)
        cylinders = st.number_input("Cylinders", step=1)
        horsepower = st.number_input("HorsePower", step=5)
        weight = st.number_input("Weight (lbs)", step=100)
        wheelbase = st.number_input("Wheelbase (in)", step=1)
        length = st.number_input("Length (in)", step=1)
        mpg_avg = st.number_input("MPG Average", step=1)

    actual_msrp = st.number_input("Optional: Actual MSRP (if known)", step=500.0)

    if st.button("🔁 Predict + Update Model"):
        new_data = pd.DataFrame([[brand, model_name, vehicle_class, region, drivetrain,
                                  dealer_cost, engine_size, cylinders, horsepower,
                                  weight, wheelbase, length, mpg_avg]],
                                columns=['Brand', 'Model', 'VehicleClass', 'Region', 'DriveTrain',
                                         'DealerCost', 'EngineSize', 'Cylinders', 'HorsePower',
                                         'Weight', 'Wheelbase', 'Length', 'MPG_Avg'])

        # If actual MSRP is known, use it to retrain the model
        if actual_msrp > 0:
            new_data['MSRP'] = actual_msrp
            data = pd.concat([data, new_data], ignore_index=True)
            joblib.dump(data, 'data.joblib')  # Save updated dataset
            model = train_model(data)         # Retrain model with new data
            st.success("✅ Model retrained with new data!")

        # Predict MSRP
        try:
            result = model.predict(new_data)[0]
            st.success(f"💰 Predicted MSRP: **${result:,.2f}**")
        except Exception as e:
            st.error(f"⚠️ Prediction failed: {e}")

elif page == "📊 EDA Insights":
    st.header("📊 Exploratory Data Analysis")

    if data.empty:
        st.warning("No data available yet.")
    else:
        st.subheader("🔗 Correlation Heatmap")
        numerical_columns = ['MSRP', 'DealerCost', 'EngineSize', 'Cylinders', 'HorsePower',
                             'Weight', 'Wheelbase', 'Length', 'MPG_Avg']
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(data[numerical_columns].corr(), annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)

        st.subheader("📦 Boxplot: MSRP by Vehicle Class")
        fig2, ax2 = plt.subplots(figsize=(10, 5))
        sns.boxplot(data=data, x='VehicleClass', y='MSRP', ax=ax2)
        ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45)
        st.pyplot(fig2)

        st.subheader("📈 Distribution of Numerical Features")
        selected_col = st.selectbox("Choose a feature to visualize", numerical_columns)
        fig3, ax3 = plt.subplots(figsize=(8, 4))
        sns.histplot(data[selected_col], kde=True, bins=30, ax=ax3)
        ax3.set_title(f"Distribution of {selected_col}")
        st.pyplot(fig3)

        st.subheader("📄 Raw Data Preview")
        st.dataframe(data.tail(10))
