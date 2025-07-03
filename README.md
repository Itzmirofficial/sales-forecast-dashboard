# SalesML – Advanced Sales Forecasting Dashboard

**SalesML** is a fully interactive, machine learning–powered dashboard designed to forecast store-item level sales using historical data. Built with **Streamlit**, **XGBoost**, and **Plotly**, it enables supply chain professionals and data enthusiasts to visualize trends, analyze prediction errors, and export forecast data with ease.

---

## 🔗 Live App

Access the deployed application here:  
👉 [https://salesml.streamlit.app](https://salesml.streamlit.app)

---

## 📌 Project Highlights

- Forecast daily sales using an XGBoost Regressor
- Visualize actual vs predicted sales with interactive line charts
- Track model performance using RMSE, MAE, R², and MSE
- Analyze feature importance and rolling averages
- Examine residual errors with histograms and time plots
- Identify high-error (outlier) dates
- View sales heatmaps by month and weekday
- Export forecasts as downloadable CSV files
- View an auto-generated, professional summary report

---

## ⚙️ Tech Stack

- **Frontend/UI**: Streamlit
- **Machine Learning**: XGBoost
- **Data Handling**: Pandas, NumPy
- **Visualization**: Plotly, Seaborn, Matplotlib
- **Deployment**: Streamlit Cloud

---

## 📁 Directory Structure

```
sales-forecast-dashboard/
├── app.py              # Streamlit app
├── train.csv           # Sales data (required to run locally)
├── requirements.txt    # Dependencies
└── README.md          # Project overview
```

---

## 🚀 Getting Started

### Step 1: Clone the Repository

```bash
git clone https://github.com/itzmirofficial/sales-forecast-dashboard.git
cd sales-forecast-dashboard
```

### Step 2: Install Dependencies

Make sure you have Python 3.9+ installed.

```bash
pip install -r requirements.txt
```

### Step 3: Run the Streamlit App

```bash
streamlit run app.py
```

The dashboard will launch in your browser at `http://localhost:8501`.

---

## 📊 Visual Insights

The dashboard includes the following visualizations and insights:

- **Actual vs Predicted Sales** – Interactive line chart
- **Feature Importance** – Bar chart from trained XGBoost model
- **Residual Distribution** – Histogram of prediction errors
- **Error Over Time** – Time series of residuals
- **Rolling Averages** – Sales with 3-day and 7-day smoothing
- **Heatmap** – Average sales by month and weekday
- **High Error Dates** – Flagged outliers in prediction residuals

---

## 📤 Export Options

- Download forecast results as a CSV file
- All charts are interactive and suitable for presentations

---

## 🧑‍💻 Developed By

**Mir Abdul Aziz Khan**  
Data Science | Full-Stack Engineering  
GitHub: [@itzmirofficial](https://github.com/itzmirofficial)  
Email: khanmirabdulaziz2@gmail.com
---

## 🧠 Use Cases

- Retail demand forecasting
- Inventory and warehouse planning
- Time series performance analysis
- Executive decision-making dashboards

---

## 🪪 License

This project is licensed under the MIT License.

---

## 🙏 Acknowledgements

- [Streamlit](https://streamlit.io/)
- [XGBoost](https://xgboost.readthedocs.io/)
- [Plotly](https://plotly.com/)
- [Kaggle - Favorita Grocery Sales Forecasting](https://www.kaggle.com/c/favorita-grocery-sales-forecasting)

---

## ✅ Deployment Checklist

- ✅ Clean, production-ready codebase
- ✅ Compatible requirements.txt
- ✅ Stable dependency versions
- ✅ Streamlit Cloud deployment
- ✅ Dataset validation and error handling
