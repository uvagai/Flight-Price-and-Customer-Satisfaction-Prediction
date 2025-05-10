#!/usr/bin/env python
# coding: utf-8

# In[6]:


import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pandas as pd

# Load your customer satisfaction data
# Assuming 'customer_satisfaction_data.csv' is the data file you have
df = pd.read_csv("cleaned_passenger.csv")

# Prepare the features (X) and target (y)
X = df.drop(columns=["satisfaction"])  # Features (everything except the target)
y = df["satisfaction"]  # Target column

# Define the columns (features) for preprocessing
categorical_cols = ['Customer_Type', 'Type_of_Travel', 'Class']  # Example
numerical_cols = ['Age', 'Flight_Distance']  # Example

# Preprocessing for categorical and numerical columns
categorical_transformer = OneHotEncoder(handle_unknown='ignore')
numerical_transformer = StandardScaler()

# Create a full pipeline that first preprocesses data, then trains a model
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', categorical_transformer, categorical_cols),
        ('num', numerical_transformer, numerical_cols)
    ]
)

# Build the full pipeline
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier())
])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model_pipeline.fit(X_train, y_train)

# Save the trained model
joblib.dump(model_pipeline, 'customer_satisfaction_pipeline.pkl')

print("Model trained and saved as 'customer_satisfaction_pipeline.pkl'")


# In[ ]:




