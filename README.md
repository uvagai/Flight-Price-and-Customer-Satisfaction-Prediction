# Flight Price and Customer Satisfaction Prediction

## Project 1: Flight Price Prediction (Regression)

**Domain**: Travel & Tourism  
**Skills**: Python, Streamlit, Machine Learning, Data Analysis, MLflow  
**Tech Stack**: Pandas, Scikit-learn, XGBoost, Joblib, MLflow, Streamlit  

### Problem Statement:
Designed an end-to-end machine learning solution to predict flight ticket prices based on various travel parameters, including airline, route, date, and time. The goal was to assist users in planning travel more efficiently and help travel businesses optimize pricing strategies.

### Approach:
- **Data Preprocessing**:
  - Cleaned and transformed raw data (`Flight_Price.csv`) by parsing dates, computing duration, and handling categorical variables.
- **Feature Engineering**:
  - Extracted features like `Journey_Day`, `Journey_Month`, `Dep_hour`, and `Arr_min` from timestamps.
- **Model Building**:
  - Built regression models using Linear Regression, Random Forest, and XGBoost.
  - Evaluated using RMSE and R².
- **Experiment Tracking**:
  - Logged metrics, models, and parameters using **MLflow**.
- **Deployment**:
  - Developed an interactive **Streamlit** app that takes user inputs and predicts flight prices in real-time.

### Outcome:
- Achieved high-accuracy predictions with XGBoost regression.
- Successfully deployed the model in a dynamic UI with MLflow integration for model tracking.

---

## Project 2: Customer Satisfaction Prediction (Classification)

**Domain**: Customer Experience  
**Skills**: Python, Streamlit, Machine Learning, Classification, Data Analysis, MLflow  
**Tech Stack**: Pandas, Scikit-learn, Joblib, MLflow, Streamlit  

### Problem Statement:
Built a machine learning classifier to predict airline customer satisfaction based on demographic and service quality attributes. The project aimed to help airlines improve customer experience and retention strategies.

### Approach:
- **Data Preprocessing**:
  - Cleaned and prepared the `Passenger_Satisfaction.csv` dataset.
  - Handled missing values and encoded categorical features.
- **EDA & Feature Engineering**:
  - Explored trends and feature correlations using visual analytics.
- **Model Building**:
  - Trained models using Logistic Regression, Random Forest, and Gradient Boosting.
  - Evaluated with metrics like Accuracy, F1-score, and Confusion Matrix.
- **Experiment Tracking**:
  - Logged models and performance data using **MLflow**.
- **Deployment**:
  - Developed a **Streamlit** app where users can input customer attributes and get real-time satisfaction predictions.

### Outcome:
- Achieved high classification accuracy with Random Forest and Gradient Boosting.
- Delivered an end-to-end predictive tool embedded in a user-friendly Streamlit interface, fully tracked with MLflow.

---

## Project Deliverables:
- Python scripts for data preprocessing, model training, and MLflow integration.
- Clean CSV files containing processed flight and customer data.
- Regression and classification models for price prediction and customer satisfaction.
- A Streamlit app for data visualization and prediction with MLflow metadata integration.
- Documentation covering methodology, analysis, and insights.

---

## Dataset Links:
- **Flight Price Dataset**: [Flight_Price.csv](link-to-dataset)
- **Customer Satisfaction Dataset**: [Passenger_Satisfaction.csv](link-to-dataset)

---

## Evaluation Metrics:
- **Flight Price Prediction**:
  - RMSE (Root Mean Squared Error)
  - R² (Coefficient of Determination)
  
- **Customer Satisfaction Prediction**:
  - Accuracy
  - F1-score
  - Confusion Matrix

---

## Technical Tags:
- Python, Data Cleaning, Feature Engineering, Machine Learning, Regression, Classification, Streamlit, MLflow

