# 🚗 Car Price Prediction App

A complete end-to-end Machine Learning project that predicts used car prices based on multiple vehicle attributes such as brand, model, year, engine type, mileage, and more.

This project includes data preprocessing, feature engineering, model training, hyperparameter optimization, and a Streamlit web interface for real-time predictions.

## ⭐ Key Features

- 📊 **Full ML pipeline** — From raw data extraction to preprocessing, modeling, and optimization.
- 🧹 **Data cleaning & transformation** — Robust handling of missing values, duplicates, and formatting.
- 🤖 **Optimized Ridge Regression model** — Selected via rigorous testing against other regression algorithms.
- 📈 **Performance metrics** — Evaluated using MAE, RMSE, and R² scores.
- 🌐 **Streamlit app** — A user-friendly web interface for real-time price estimation.
- 💾 **Joblib model pipeline** — Efficient model serialization for deployment.

## 📁 Project Structure

```
Car_Price_Prediction/
│
├── app.py                            # Streamlit application entry point
│
├── model/
│   └── final_ridge_pipeline.joblib   # Final trained ML model pipeline
│
├── notebook/
│   └── car_price_model_training.ipynb # Jupyter notebook (Complete ML workflow)
│
├── data/
│   └── Large Cars Dataset.csv        # Dataset used for training
│
├── requirements.txt                  # Python dependencies
├── .gitignore                        # Files to ignore in version control
└── README.md                         # Project documentation
```

## 🧠 Machine Learning Workflow

The project follows a structured data science lifecycle:

### 1. Data Exploration & Cleaning

- **Duplicate Removal**: Eliminated duplicate entries to prevent bias.
- **Missing Value Handling**: Imputed or removed null values based on column significance.
- **Standardization**: Corrected data formats (strings to numbers, date parsing).
- **EDA**: Checked correlations and distributions to understand data variance.

### 2. Feature Engineering

- **Categorical Encoding**: Applied Label Encoding/One-Hot Encoding for non-numeric features.
- **Scaling**: Standardized numeric variables to bring them to a common scale.
- **Outlier Handling**: Detected and managed outliers to improve model stability.
- **Feature Selection**: Identified the most impactful variables for price prediction.

### 3. Model Training & Optimization

- Trained multiple regression models (Linear, Lasso, Ridge, Random Forest).
- **Ridge Regression** was selected as the best performer.
- Applied **Hyperparameter Tuning** to find the optimal alpha values.
- **Evaluation Metrics**:
  - Mean Absolute Error (MAE)
  - Root Mean Squared Error (RMSE)
  - R-squared (R²)

### 4. Model Saving

The full pipeline (preprocessing + model) is saved using Joblib:
```
model/final_ridge_pipeline.joblib
```

### 5. Deployment Using Streamlit

The `app.py` script loads the saved pipeline and renders a simple UI where users can select car details and get an instant price prediction.

## 🚀 How to Run the Project

Follow these steps to set up the project locally:

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd Car_Price_Prediction
```

### 2. Install dependencies

Ensure you have Python installed, then run:

```bash
pip install -r requirements.txt
```

### 3. Run the application

Launch the Streamlit interface:

```bash
streamlit run app.py
```

The app will open automatically in your default browser at `http://localhost:8501`.

## 📓 Jupyter Notebook

For a deep dive into the code, analysis, and visualization, check the notebook:
```
notebook/car_price_model_training.ipynb
```

## 🛠 Technologies Used

- **Language**: Python
- **Data Manipulation**: NumPy, Pandas
- **Machine Learning**: Scikit-Learn
- **Visualization**: Matplotlib, Seaborn
- **Web Framework**: Streamlit
- **Serialization**: Joblib

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!
Feel free to open an issue or submit a pull request if you have ideas for improvements.

## 📜 License

This project is licensed under the MIT License.

## ⭐ Support

If you found this project helpful, please consider giving it a star ⭐ on GitHub!