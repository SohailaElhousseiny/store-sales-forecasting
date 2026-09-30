import streamlit as st
import pandas as pd
import joblib
import base64

# =========================================================
# Page Configuration
# =========================================================

st.set_page_config(
    page_title="Store Sales Forecasting",
    page_icon="📊",
    layout="wide"
)

# =========================================================
# Background Image
# =========================================================

def set_background(image_file):

    with open(image_file, "rb") as image:
        encoded_image = base64.b64encode(
            image.read()
        ).decode()

    st.markdown(
        f"""
        <style>

        /* Main Background */
        .stApp {{
            background-image:
                linear-gradient(
                    rgba(8, 15, 30, 0.78),
                    rgba(8, 15, 30, 0.78)
                ),
                url("data:image/jpeg;base64,{encoded_image}");

            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}

        /* Main Content */
        .main {{
            background: transparent;
        }}

        /* General Text */
        .stApp,
        .stMarkdown,
        p,
        label,
        span {{
            color: #F1F5F9;
        }}

        /* Main Title */
        h1 {{
            color: #FFFFFF !important;
            font-size: 42px !important;
            font-weight: 700 !important;
        }}

        /* Section Titles */
        h2, h3 {{
            color: #FFFFFF !important;
            font-weight: 650 !important;
        }}

        /* Metric Cards */
        div[data-testid="stMetric"] {{
            background: rgba(15, 23, 42, 0.88);
            padding: 20px;
            border-radius: 15px;
            border: 1px solid rgba(148, 163, 184, 0.25);
            box-shadow: 0 5px 20px rgba(0, 0, 0, 0.30);
        }}

        div[data-testid="stMetric"] label {{
            color: #CBD5E1 !important;
        }}

        div[data-testid="stMetricValue"] {{
            color: #FFFFFF !important;
        }}

        /* Sidebar */
        section[data-testid="stSidebar"] {{
            background: rgba(8, 15, 30, 0.96);
        }}

        section[data-testid="stSidebar"] * {{
            color: #F1F5F9 !important;
        }}

        /* Number Inputs */
        div[data-baseweb="input"] {{
            background: rgba(30, 41, 59, 0.95);
            border-radius: 8px;
        }}

        /* Select Box */
        div[data-baseweb="select"] > div {{
            background: rgba(30, 41, 59, 0.95);
        }}

        /* Prediction Button */
        .stButton > button {{
            width: 100%;
            height: 50px;
            border-radius: 10px;
            background: #2563EB;
            color: white;
            font-size: 17px;
            font-weight: 600;
            border: none;
        }}

        .stButton > button:hover {{
            background: #1D4ED8;
            color: white;
        }}

        /* Info and Success Boxes */
        div[data-testid="stAlert"] {{
            border-radius: 10px;
        }}

        /* Dividers */
        hr {{
            border-color: rgba(148, 163, 184, 0.25);
        }}

        </style>
        """,
        unsafe_allow_html=True
    )


set_background("sales_background.jpg")

# =========================================================
# Load Model
# =========================================================

model = joblib.load("sales_model.pkl")
features = joblib.load("features.pkl")

# =========================================================
# Load Historical Data
# =========================================================

df = pd.read_csv(
    "stores_sales_forecasting.csv",
    encoding="latin1"
)

df["Order Date"] = pd.to_datetime(df["Order Date"])

monthly_sales = (
    df.resample("MS", on="Order Date")["Sales"]
    .sum()
    .reset_index()
)

# =========================================================
# Header
# =========================================================

st.title("📊 Store Sales Forecasting")

st.markdown(
    "### Machine Learning Dashboard for Monthly Sales Prediction"
)

st.write(
    "Analyze historical store sales and predict future monthly sales "
    "using a Random Forest Regression model."
)

st.divider()

# =========================================================
# KPI Section
# =========================================================

total_sales = monthly_sales["Sales"].sum()

average_sales = monthly_sales["Sales"].mean()

highest_sales = monthly_sales["Sales"].max()

highest_month = monthly_sales.loc[
    monthly_sales["Sales"].idxmax(),
    "Order Date"
]

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "💰 Total Sales",
        f"${total_sales:,.0f}"
    )

with col2:
    st.metric(
        "📊 Average Monthly Sales",
        f"${average_sales:,.0f}"
    )

with col3:
    st.metric(
        "🏆 Highest Monthly Sales",
        f"${highest_sales:,.0f}"
    )

with col4:
    st.metric(
        "📅 Best Sales Month",
        highest_month.strftime("%b %Y")
    )

st.divider()

# =========================================================
# Historical Sales Chart
# =========================================================

st.subheader("📈 Historical Monthly Sales")

st.write(
    "Monthly sales performance based on historical transaction data."
)

chart_data = monthly_sales.set_index(
    "Order Date"
)["Sales"]

st.line_chart(chart_data)

st.divider()

# =========================================================
# Forecast Section
# =========================================================

st.subheader("🔮 Sales Forecast")

st.write(
    "Enter the forecast period and the previous three months of sales."
)

col1, col2 = st.columns(2)

# =========================================================
# Forecast Date
# =========================================================

with col1:

    year = st.number_input(
        "Forecast Year",
        min_value=2018,
        max_value=2035,
        value=2026,
        step=1,
        key="forecast_year"
    )

    month = st.selectbox(
        "Forecast Month",
        list(range(1, 13)),
        format_func=lambda x: pd.Timestamp(
            year=2000,
            month=x,
            day=1
        ).strftime("%B"),
        key="forecast_month"
    )

# =========================================================
# Previous Sales
# =========================================================

with col2:

    lag_1 = st.number_input(
        "Previous Month Sales",
        min_value=0.0,
        value=10000.0,
        step=500.0,
        key="previous_month_sales"
    )

    lag_2 = st.number_input(
        "Sales 2 Months Ago",
        min_value=0.0,
        value=10000.0,
        step=500.0,
        key="sales_two_months_ago"
    )

    lag_3 = st.number_input(
        "Sales 3 Months Ago",
        min_value=0.0,
        value=10000.0,
        step=500.0,
        key="sales_three_months_ago"
    )

# =========================================================
# Quarter Calculation
# =========================================================

quarter = ((month - 1) // 3) + 1

st.info(
    f"📅 Selected Quarter: *Q{quarter}*"
)

# =========================================================
# Prediction Button
# =========================================================

predict_button = st.button(
    "🔮 Predict Sales",
    use_container_width=True,
    key="predict_sales_button"
)

# =========================================================
# Prediction
# =========================================================

if predict_button:

    input_data = pd.DataFrame({
        "Year": [year],
        "Month": [month],
        "Quarter": [quarter],
        "Lag_1": [lag_1],
        "Lag_2": [lag_2],
        "Lag_3": [lag_3]
    })

    prediction = model.predict(
        input_data[features]
    )[0]

    st.divider()

    st.subheader("🎯 Prediction Result")

    month_name = pd.Timestamp(
        year=year,
        month=month,
        day=1
    ).strftime("%B")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Forecast Period",
            f"{month_name} {year}"
        )

    with col2:
        st.metric(
            "Quarter",
            f"Q{quarter}"
        )

    with col3:
        st.metric(
            "Predicted Sales",
            f"${prediction:,.2f}"
        )

    st.success(
        f"📈 Estimated sales for *{month_name} {year}*: "
        f"*${prediction:,.2f}*"
    )

# =========================================================
# About the Project
# =========================================================

st.divider()

st.subheader("ℹ️ About This Project")

col1, col2, col3 = st.columns(3)

with col1:

    st.write("### 📂 Dataset")

    st.write(
        "Historical store sales transactions"
    )

with col2:

    st.write("### 🤖 Machine Learning Model")

    st.write(
        "Random Forest Regression"
    )

with col3:

    st.write("### 🎯 Prediction Target")

    st.write(
        "Monthly Sales"
    )

# =========================================================
# Model Features
# =========================================================

st.divider()

st.subheader("🧠 Model Features")

st.write(
    "The model uses the following features to estimate monthly sales:"
)

feature_col1, feature_col2, feature_col3 = st.columns(3)

with feature_col1:
    st.info("📅 Year")

with feature_col2:
    st.info("📆 Month")

with feature_col3:
    st.info("🗓️ Quarter")

feature_col1, feature_col2, feature_col3 = st.columns(3)

with feature_col1:
    st.info("📈 Previous Month Sales")

with feature_col2:
    st.info("📊 Sales 2 Months Ago")

with feature_col3:
    st.info("📉 Sales 3 Months Ago")

# =========================================================
# Footer
# =========================================================

st.divider()

st.caption(
    "Store Sales Forecasting • Machine Learning Portfolio Project"
)

