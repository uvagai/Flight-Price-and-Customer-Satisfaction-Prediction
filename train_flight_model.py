#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
import joblib

# 1. Load your cleaned dataset
df = pd.read_csv("Cleaned_Flight_Price.csv")  # Use your cleaned CSV file

# 2. Define features and target
X = df.drop("Price", axis=1)
y = df["Price"]

# 3. Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Identify categorical and numerical features
categorical_cols = ["Airline", "Source", "Destination", "Total_Stops"]
numerical_cols = ["Journey_Day", "Journey_Month", "Dep_hour", "Dep_min", "Arr_hour", "Arr_min", "Duration_hours"]

# 5. Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
    ],
    remainder='passthrough'  # pass numerical columns
)

# 6. Define model pipeline
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42))
])

# 7. Train the pipeline
model.fit(X_train, y_train)

# 8. Evaluate
y_pred = model.predict(X_test)
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R² Score:", r2_score(y_test, y_pred))

# 9. Save the pipeline
joblib.dump(model, "flight_price_pipeline.pkl")
print("✅ Model saved as flight_price_pipeline.pkl")


# In[ ]:




