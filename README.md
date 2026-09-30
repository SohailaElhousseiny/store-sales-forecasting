
📊 Store Sales Forecasting

A Machine Learning project that analyzes historical store sales data and predicts future monthly sales using Random Forest Regression.

🎯 Project Overview

Sales forecasting helps businesses understand their historical performance and estimate future sales.

This project transforms transaction-level sales data into monthly sales data, creates time-based and lag features, trains a Machine Learning regression model, and provides an interactive Streamlit dashboard for exploring historical sales and generating predictions.

---

💡 Business Problem

Businesses need to estimate future sales to support decisions related to:

- Inventory planning
- Sales planning
- Business forecasting
- Performance analysis
- Resource allocation

The goal of this project is to build a simple Machine Learning solution that can estimate monthly sales based on historical sales patterns.

---

📂 Dataset

The dataset contains historical store sales transactions.

Dataset Information

- Rows: 2,121
- Columns: 21
- Date Range: January 2014 – December 2017
- Target: Monthly Sales

Main Dataset Columns

- Order Date
- Ship Date
- Ship Mode
- Customer ID
- Customer Name
- Segment
- Country
- City
- State
- Region
- Category
- Sub-Category
- Product Name
- Sales
- Quantity
- Discount
- Profit

---

🔄 Data Preprocessing

The following preprocessing steps were performed:

1. Loaded the CSV dataset using Pandas.
2. Converted "Order Date" into a datetime format.
3. Aggregated transaction-level sales into monthly sales.
4. Extracted time-based features from the date.
5. Created lag features based on previous monthly sales.
6. Removed rows with missing lag values.
7. Split the data chronologically into training and testing sets.

---

🧠 Feature Engineering

The model uses the following features:

Feature| Description
Year| Year of the forecast period
Month| Month of the forecast period
Quarter| Quarter of the year
Lag_1| Sales from the previous month
Lag_2| Sales from two months ago
Lag_3| Sales from three months ago

These features allow the model to learn both seasonal patterns and recent sales trends.

---

🤖 Machine Learning Model

The project uses:

Random Forest Regressor

Model Configuration

RandomForestRegressor(
    n_estimators=200,
    max_depth=5,
    random_state=42
)

The data was split chronologically:

- Training samples: 36
- Testing samples: 9

A chronological split was used instead of a random split because this is a time-based forecasting problem.

---

📊 Model Evaluation

The model was evaluated using Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE).

Metric| Result
MAE| 4,248.91
RMSE| 4,955.69

MAE

MAE measures the average absolute difference between the predicted sales and the actual sales.

RMSE

RMSE measures prediction error while giving larger errors more weight.

---

🌐 Streamlit Dashboard

The project includes an interactive Streamlit dashboard.

The dashboard provides:

- 📊 Total Sales
- 📈 Average Monthly Sales
- 🏆 Highest Monthly Sales
- 📅 Best Sales Month
- 📉 Historical Monthly Sales Chart
- 🔮 Future Sales Prediction
- 📋 Model Feature Information

Users can enter:

- Forecast Year
- Forecast Month
- Previous Month Sales
- Sales from Two Months Ago
- Sales from Three Months Ago

The application then generates an estimated monthly sales value using the trained Random Forest model.

---

🖥️ Project Structure

stores_sales/
│
├── app.py
├── model.py
├── stores_sales_forecasting.csv
├── sales_model.pkl
├── features.pkl
├── sales_background.jpg
├── requirements.txt
└── README.md

---

⚙️ Installation

1. Clone the repository

git clone YOUR_GITHUB_REPOSITORY_URL

2. Navigate to the project folder

cd stores_sales

3. Create a virtual environment

python -m venv venv

4. Activate the virtual environment

Windows

venv\Scripts\activate

5. Install the required libraries

pip install -r requirements.txt

---

🚀 Run the Application

Start the Streamlit dashboard using:

python -m streamlit run app.py

The application will open in your browser.

---

📌 Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Joblib
- Streamlit
- Matplotlib / Streamlit Charts

---

🔮 Future Improvements

Possible future improvements include:

- Testing additional forecasting models
- Hyperparameter tuning
- Adding more historical lag features
- Adding rolling averages
- Adding interactive filters
- Adding product and regional sales forecasting
- Deploying the application online
- Building multi-step future forecasting

---

👩‍💻 Project Purpose

This project was developed as a Machine Learning portfolio project to demonstrate skills in:

- Data preprocessing
- Exploratory data analysis
- Feature engineering
- Time-based data handling
- Regression modeling
- Model evaluation
- Streamlit application development
- Machine Learning deployment concepts

---

📜 License

This project is intended for educational and portfolio purposes.
