#!/usr/bin/env python
# coding: utf-8

# In[108]:


import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import pickle
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_squared_error
import datetime as dt
from sklearn.preprocessing import LabelEncoder


# In[110]:


import pandas as pd
import numpy as np

# Load dataset
df = pd.read_csv("Flight_Price.csv")

# Show initial data
print("Shape of data:", df.shape)
df.head()


# In[112]:


df.dropna(inplace=True)


# In[114]:


# Check for null values
print("Missing values:\n", df.isnull().sum())

# Check data types
df.info()


# In[116]:


duplicate_rows = df.duplicated()
print("Number of duplicate rows:", duplicate_rows.sum())


# In[118]:


df = df.drop_duplicates()


# In[120]:


# Show available columns to verify actual names
print("Available columns:", df.columns.tolist())

# Try dropping only if the columns exist
columns_to_drop = ['Route', 'Additional_Info']
existing_cols_to_drop = [col for col in columns_to_drop if col in df.columns]

# Drop the columns if they exist
df.drop(existing_cols_to_drop, axis=1, inplace=True)



# In[122]:


print(df.columns)


# Convert date, time colmns

# In[125]:


df['Journey_Day'] = pd.to_datetime(df['Date_of_Journey'],format='%d/%m/%Y').dt.day                                 
df['Journey_Month'] = pd.to_datetime(df['Date_of_Journey'],format='%d/%m/%Y').dt.month
df.drop('Date_of_Journey',inplace=True, axis=1 )


# In[127]:


df.head(2)


# In[129]:


df['Dep_hour'] = pd.to_datetime(df['Dep_Time'], format='%H:%M').dt.hour
df['Dep_min'] = pd.to_datetime(df['Dep_Time'],format='%H:%M').dt.minute
df.drop('Dep_Time', inplace=True, axis=1)


# In[131]:


df['Arrival_Time'] = df['Arrival_Time'].str.split(" ").str[0]
df['Arr_hour'] = pd.to_datetime(df['Arrival_Time'],format='%H:%M').dt.hour
df['Arr_min'] = pd.to_datetime(df['Arrival_Time'],format='%H:%M').dt.minute
df.drop('Arrival_Time', inplace=True, axis=1)

df.head(3)


# In[133]:


df['Duration_hours']=  df['Duration'].str.replace("h", '*1').str.replace(' ','+').str.replace('m','/60').apply(eval)


# In[135]:


df.drop(['Duration'], axis =1, inplace = True)


# In[137]:


df['Duration_hours'] = df['Duration_hours'].astype(float).round(3)


# In[139]:


df.head(2)


# In[141]:


df['Total_Stops'].unique()


# In[143]:


df['Total_Stops'] = df['Total_Stops'].map({
'non-stop': 0,
'1 stop': 1,
'2 stops': 2,
'3 stops': 3,
'4 stops': 4})


# In[145]:


df['Total_Stops'].unique()


# In[147]:


df

# Get the column with the highest value for each row to create a single 'Airline' column
X['Airline'] = X[['Airline_Air India', 'Airline_GoAir', 'Airline_IndiGo', 'Airline_Jet Airways',
                  'Airline_Jet Airways Business', 'Airline_Multiple carriers',
                  'Airline_Multiple carriers Premium economy', 'Airline_SpiceJet',
                  'Airline_Trujet', 'Airline_Vistara', 'Airline_Vistara Premium economy']].idxmax(axis=1)

# Now you can apply label encoding to the 'Airline' column
label_encoder = LabelEncoder()
X['Airline'] = label_encoder.fit_transform(X['Airline'])
# Get the column with the highest value for each row to create a single 'Airline' column
X['Source'] = X[['Source_Chennai', 'Source_Delhi', 'Source_Kolkata', 'Source_Mumbai']].idxmax(axis=1)

# Now you can apply label encoding to the 'Airline' column
label_encoder = LabelEncoder()
X['Source'] = label_encoder.fit_transform(X['Source'])
# Get the column with the highest value for each row to create a single 'Airline' column
X['Destination'] = X[['Destination_Cochin', 'Destination_Delhi', 'Destination_Hyderabad',
       'Destination_Kolkata', 'Destination_New Delhi']].idxmax(axis=1)

# Now you can apply label encoding to the 'Airline' column
label_encoder = LabelEncoder()
X['Destination'] = label_encoder.fit_transform(X['Destination'])
#from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()

# Apply label encoding to columns like 'Airline', 'Source', 'Destination', etc.
X['Airline'] = label_encoder.fit_transform(X['Airline'])
X['Source'] = label_encoder.fit_transform(X['Source'])
X['Destination'] = label_encoder.fit_transform(X['Destination'])

# In[ ]:





# In[150]:


print(df.columns)


# In[152]:


# Save cleaned dataset
df.to_csv("Cleaned_Flight_Price.csv", index=False)

print("✅ Data preprocessing complete! Cleaned dataset saved.")


# In[154]:


#Apply Label Encoding to all object (categorical) columns
label_encoders = {}
for column in df.select_dtypes(include='object').columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

# 3. Save encoders for use in Streamlit
with open("label_encoders.pkl", "wb") as f:
    
    pickle.dump(label_encoders, f)


# In[156]:


df_train = pd.get_dummies(df, columns=['Airline','Source','Destination'], drop_first=True, dtype=int)


# In[158]:


X = df_train.drop('Price', axis='columns')
y = df_train['Price']


# Exploratory data  analysis

# In[161]:


sns.histplot(df['Price'], kde=True)
plt.show()


# In[57]:


sns.boxplot(df['Price'])
plt.show()


# In[59]:


df['Airline'].value_counts()


# In[61]:


fig = px.box(df, x = 'Airline' , y = 'Price')
fig.show()


# In[63]:


sns.barplot(df, x = 'Total_Stops' , y = 'Price', width = 0.4)
plt.show()


# In[169]:


cm = df_train.corr()
plt.figure(figsize=(20,12))
sns.heatmap(cm, annot=True)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()


# Data standardization / Normalization

# In[66]:


df_train = pd.get_dummies(df, columns=['Airline','Source','Destination'], drop_first=True, dtype=int)


# In[68]:


X = df_train.drop('Price', axis='columns')
y = df_train['Price']


# In[70]:


df_train.head(3)


# split data and train regression
# 
pip install xgboost

# In[73]:


pip install mlflow


# In[75]:


from sklearn.model_selection import train_test_split

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[77]:


# Check for missing values in your dataset
print(X.isnull().sum())


# In[79]:


# Check again for any missing values
print(X_train.isnull().sum())


# In[81]:


# Print column names to check the exact name of 'Total_Stops'
print(X_train.columns)


# In[ ]:





# In[84]:


# Importing necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb

# Assuming your data is in a DataFrame 'df' and the target variable is 'Price'

# Feature selection (X) and target variable (y)
X = df_train.drop('Price', axis='columns') # Drop the 'Price' column from the features
y = df_train['Price']  # The target variable is 'Price'

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# shape of the X_train, X_test, y_train, y_test features
print("x train: ",X_train.shape)
print("x test: ",X_test.shape)
print("y train: ",y_train.shape)
print("y test: ",y_test.shape)
# Linear Regression Model
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
y_pred_lin_reg = lin_reg.predict(X_test)
mse_lin_reg = mean_squared_error(y_test, y_pred_lin_reg)
rmse_lin_reg = mse_lin_reg**0.5
r2_lin_reg = r2_score(y_test, y_pred_lin_reg)

# Random Forest Regressor Model
rf_reg = RandomForestRegressor(n_estimators=100, random_state=42)
rf_reg.fit(X_train, y_train)
y_pred_rf_reg = rf_reg.predict(X_test)
mse_rf_reg = mean_squared_error(y_test, y_pred_rf_reg)
rmse_rf_reg = mse_rf_reg**0.5
r2_rf_reg = r2_score(y_test, y_pred_rf_reg)

# XGBoost Regressor Model
xg_reg = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)
xg_reg.fit(X_train, y_train)
y_pred_xg_reg = xg_reg.predict(X_test)
mse_xg_reg = mean_squared_error(y_test, y_pred_xg_reg)
rmse_xg_reg = mse_xg_reg**0.5
r2_xg_reg = r2_score(y_test, y_pred_xg_reg)

# Printing the results
print(f"Linear Regression - RMSE: {rmse_lin_reg}, R²: {r2_lin_reg}")
print(f"Random Forest - RMSE: {rmse_rf_reg}, R²: {r2_rf_reg}")
print(f"XGBoost - RMSE: {rmse_xg_reg}, R²: {r2_xg_reg}")


# In[85]:


print(X_train.isnull().sum())


# In[ ]:





# In[89]:


import mlflow
import mlflow.sklearn
import mlflow.xgboost

models = [
    ("Linear Regression", 
        {'n_jobs':1, 'positive':False},
        LinearRegression(), 
        (X_train, y_train),
        (X_test, y_test)
    ),
    (
        "Random Forest Regressor", 
        {'n_estimators': 1100, 'min_samples_split': 10, 'min_samples_leaf': 2, 'max_depth': 30},
        RandomForestRegressor(), 
        (X_train, y_train),
        (X_test, y_test)
    ),
    (
        "XGBRegressor",
        {'n_estimators': 1100, 'learning_rate': 0.05},
        XGBRegressor(), 
        (X_train, y_train),
        (X_test, y_test)
    )
]
# In[91]:


print(X_train.isnull().sum())

# Check for infinite values
print(np.any(np.isinf(X_train)))
print(np.any(np.isinf(y_train)))

# Replace infinity values with NaN and then handle them
X_train = np.where(np.isinf(X_train), np.nan, X_train)
y_train = np.where(np.isinf(y_train), np.nan, y_train)

# In[97]:


print(df.columns)

reports = []

for model_name, params, model, train_set, test_set in models:
    X_train = train_set[0]
    y_train = train_set[1]
    X_test = test_set[0]
    y_test = test_set[1]
    
    model.set_params(**params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test,y_pred))
    r2 = r2_score(y_test,y_pred)
    reports.append((model_name,rmse,r2))
# In[164]:


import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score

# Step 1: Set MLflow Tracking URI and Experiment Name
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("Flight Price Prediction")

# Step 2: RandomizedSearchCV for Random Forest
param_distributions = {
    'max_depth': [5, 10, 15, 20, 25, 30],
    'min_samples_leaf': [1, 2, 5, 10],
    'min_samples_split': [2, 5, 10, 15, 100],
    'n_estimators': [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200]
}

rf_random = RandomizedSearchCV(
    estimator=RandomForestRegressor(),
    param_distributions=param_distributions,
    n_iter=6,
    scoring='r2',
    cv=5,
    verbose=1,
    n_jobs=-1,
    random_state=42
)

# Fit on training data
rf_random.fit(X_train, y_train)

# Get best model and parameters
best_rf_model = rf_random.best_estimator_
best_rf_params = rf_random.best_params_

# Step 3: Define all models
models = [
    ("Linear Regression", LinearRegression(n_jobs=1, positive=False), {'n_jobs': 1, 'positive': False}),
    ("XGBRegressor", XGBRegressor(n_estimators=1100, learning_rate=0.05), {'n_estimators': 1100, 'learning_rate': 0.05}),
    ("Tuned Random Forest", best_rf_model, best_rf_params)
]

# Step 4: Train and log each model to MLflow
for name, model, params in models:
    print(f"\n🚀 Training: {name}")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Evaluation
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    # Log to MLflow
    with mlflow.start_run(run_name=name):
        mlflow.log_params(params)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2_score", r2)
        mlflow.sklearn.log_model(model, "model")
        print(f"✅ {name} logged to MLflow | RMSE: {rmse:.2f} | R²: {r2:.3f}")

mlflow.set_experiment("Flight Price Prediction")
mlflow.set_tracking_uri("http://127.0.0.1:5000")

for i, element in enumerate(models):
    model_name = element[0]
    params = element[1]
    model = element[2]
    report = reports[i]
        
    with mlflow.start_run(run_name=model_name):        
        mlflow.log_params(params)
        mlflow.log_metrics({
            'RMSE': report[1],
            'R2': report[2]
        })  
        
        if "XGB" in model_name:
            mlflow.xgboost.log_model(model, "model")
        else:
            mlflow.sklearn.log_model(model, "model")  
# In[166]:


model_name = 'Linear regression'
run_id=input('Please type RunID')
model_uri = f'runs:/{run_id}/model_name'

with mlflow.start_run(run_id=run_id):
    mlflow.register_model(model_uri=model_uri, name=model_name)


# 
# 
# linear regression
# model training

# In[100]:


import mlflow

model_name = 'Random Forest Regressor'
run_id = input('Please type RunID: ')
model_uri = f'runs:/{run_id}/model'  # Not 'model_name'

with mlflow.start_run(run_id=run_id):
    mlflow.register_model(model_uri=model_uri, name=model_name)



# In[100]:


import mlflow

# Set the correct tracking URI (if not set already)
mlflow.set_tracking_uri("http://localhost:5000")  # or your appropriate server URI

# Specify the run ID and load the model
run_id = "871106b391aa4c6683f0f270f695161b"  # Replace with the appropriate run ID
model_uri = f"runs:/{run_id}/model"

# Load the model (replace with appropriate flavor, e.g., mlflow.sklearn for sklearn models)
flight_model = mlflow.sklearn.load_model(model_uri)

# Print success message
print("Model loaded successfully.")



# In[102]:


import mlflow

run_id = "871106b391aa4c6683f0f270f695161b"  # Replace with your run ID
model_uri = f"runs:/{run_id}/model"

# Get the model's metadata
model = mlflow.pyfunc.load_model(model_uri)
print(model)


# In[106]:


pip install streamlit pandas mlflow xgboost


# In[ ]:




