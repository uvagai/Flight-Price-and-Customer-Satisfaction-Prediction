#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split


# In[3]:


df = pd.read_csv("Passenger_Satisfaction.csv")


# In[6]:


df.shape


# In[8]:


df.head(3)


# In[10]:


df.info()


# In[12]:


df.columns = df.columns.str.strip().str.replace(" ", '_')


# In[14]:


df.isna().sum()


# In[16]:


df['Arrival_Delay_in_Minutes'].describe()


# In[18]:


df['Arrival_Delay_in_Minutes'].fillna(df['Arrival_Delay_in_Minutes'].mean(), inplace=True)


# In[20]:


df.isna().sum()


# In[ ]:





# In[23]:


columns_continuous = ['Unnamed:_0','id', 'Age', 'Flight_Distance', 'Inflight_wifi_service' , 'Departure/Arrival_time_convenient', 
                      'Ease_of_Online_booking', 'Gate_location', 'Food_and_drink',	'Online_boarding',	'Seat_comfort',	
                      'Inflight_entertainment', 'On-board_service',	'Leg_room_service', 'Baggage_handling', 'Checkin_service',	
                      'Inflight_service',	'Cleanliness', 'Departure_Delay_in_Minutes', 'Arrival_Delay_in_Minutes']
columns_categorical = ['Gender', 'Customer_Type', 'Type_of_Travel',	'Class', 'satisfaction']


# In[25]:


columns_categorical 


# In[27]:


for col in columns_categorical:
    print(col, "-->", df[col].unique())


# In[29]:


df['satisfaction'] = df['satisfaction'].map({
    'satisfied': '0',
    'neutral or dissatisfied': '1'
})


# In[31]:


df['satisfaction'] = df['satisfaction'].astype(int)


# In[33]:


# detect outliers
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Assuming 'df' is your DataFrame with continuous columns
# Replace 'column1', 'column2', etc., with the actual column names you're interested in
columns_continuous = ['Unnamed:_0','id', 'Age', 'Flight_Distance', 'Inflight_wifi_service' , 'Departure/Arrival_time_convenient', 
                      'Ease_of_Online_booking', 'Gate_location', 'Food_and_drink',	'Online_boarding',	'Seat_comfort',	
                      'Inflight_entertainment', 'On-board_service',	'Leg_room_service', 'Baggage_handling', 'Checkin_service',	
                      'Inflight_service',	'Cleanliness', 'Departure_Delay_in_Minutes', 'Arrival_Delay_in_Minutes']

# Visualize the boxplot for each continuous column
for column in columns_continuous:
    plt.figure(figsize=(8, 6))
    sns.boxplot(x=df[column])
    plt.title(f'Boxplot for {column} to Detect Outliers')
    plt.xlabel(column)
    plt.show()


# __Exploratory data analysis__

# In[35]:


#age  column
df.groupby("satisfaction")['Age'].describe()


# In[38]:


df.satisfaction.value_counts()


# In[40]:


plt.figure(figsize=(8, 4))
sns.kdeplot(df['Age'][df['satisfaction'] == 0], fill=True, label='Satisfied')
sns.kdeplot(df['Age'][df['satisfaction'] == 1], fill=True, label = 'neutral or dissatisfied')
plt.title(f"Age KDE Plot with Hue by default")
plt.legend()
plt.show()


# **KDE for all the column**

# In[43]:


columns_continuous


# In[45]:


palette = {0: "violet", 1: "blue"}
plt.figure(figsize=(24, 20))  # Width, height in inches

for i, col in enumerate(columns_continuous):
    plt.subplot(6, 4, i+1)  # 1 row, 4 columns, ith subplot
    sns.kdeplot(df[col][df['satisfaction']==0], fill=True, label='Satisfied', color=palette[0])
    sns.kdeplot(df[col][df['satisfaction']==1], fill=True, label='neutral or dissatisfied', color=palette[1])
    plt.title(col)        
    plt.xlabel('')
    
plt.tight_layout()
plt.show()


# In[47]:


# Custom color palette: e.g., green for satisfied (0), red for not satisfied (1)
palette = {0: "yellow", 1: "green"}

sns.countplot(x='Gender', hue='satisfaction', data=df, palette=palette)
plt.title('Satisfaction Based on Gender')
plt.legend(title='Satisfaction (0 = Sat, 1 = Not Sat)',bbox_to_anchor=(1.05,1),loc='upper left')
plt.tight_layout()#adjest layout to fit everything
plt.show()


# In[49]:


palette = {0: "green", 1: "red"}
plt.figure(figsize=(24, 20))  # Width, height in inches

columns_categorical = ['Gender', 'Customer_Type', 'Type_of_Travel', 'Class']

for i, col in enumerate(columns_categorical):
    plt.subplot(4, 2, i+1)  # 1 row, 3 columns, ith subplot
    sns.countplot(x= col, hue='satisfaction', data=df, palette=palette)
    plt.legend(title='Satisfaction (0=Sat, 1=Not Sat)')
    plt.title(col)        
    plt.xlabel('')  
    plt.xticks(fontsize=12)

plt.tight_layout()
plt.show()


# **Remove columns that are just unique ids and don't have influence on target¶**(feature selection)

# In[52]:


df.columns


# In[54]:


df = df.drop(['Unnamed:_0', 'id'],axis="columns")
df = df.drop(['Gender'],axis="columns")


# In[56]:


df.columns


# In[58]:


df.select_dtypes(['int64', 'float64']).columns


# In[60]:


numeric_columns = ['Age', 'Flight_Distance', 'Inflight_wifi_service',
       'Departure/Arrival_time_convenient', 'Ease_of_Online_booking',
       'Gate_location', 'Food_and_drink', 'Online_boarding', 'Seat_comfort',
       'Inflight_entertainment', 'On-board_service', 'Leg_room_service',
       'Baggage_handling', 'Checkin_service', 'Inflight_service',
       'Cleanliness', 'Departure_Delay_in_Minutes', 'Arrival_Delay_in_Minutes',
       'satisfaction']


# In[62]:


plt.figure(figsize=(12, 8))
corr = df[numeric_columns].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Correlation Heatmap")
plt.show()


# __Dropped 'Departure_Delay_in_Minutes' Due to Multicollinearity with 'Arrival_Delay_in_Minutes__

# In[64]:


df = df.drop('Departure_Delay_in_Minutes', axis = 'columns')


# In[67]:


df.columns


# In[69]:


df.to_csv('passenger.output', index=False)


# In[70]:


df.to_csv('cleaned_passenger.csv', index=False)


# __Use minmax Scaler__

# In[74]:


from sklearn.preprocessing import MinMaxScaler


# In[76]:


df_train = df.copy()


# In[78]:


df_train


# In[80]:


cols = ['Age', 'Flight_Distance', 'Inflight_wifi_service',
       'Departure/Arrival_time_convenient', 'Ease_of_Online_booking',
       'Gate_location', 'Food_and_drink', 'Online_boarding', 'Seat_comfort',
       'Inflight_entertainment', 'On-board_service', 'Leg_room_service',
       'Baggage_handling', 'Checkin_service', 'Inflight_service',
       'Cleanliness', 'Arrival_Delay_in_Minutes']

scaler = MinMaxScaler()

df_train[cols] = scaler.fit_transform(df_train[cols])
df_train.describe()


# In[82]:


df_train = pd.get_dummies(df_train, columns = ['Customer_Type', 'Type_of_Travel', 'Class'], drop_first=True, dtype=int)#feature encoding


# __MODEL TRAINING__

# In[85]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

X = df_train.drop('satisfaction', axis = 1)
y = df_train['satisfaction']

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=42)

model = LogisticRegression(C = 0.1, solver='liblinear', max_iter=500, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
report = classification_report(y_test, y_pred)
print(report)


# In[86]:


feature_importance = model.coef_[0]

# Create a DataFrame for easier handling
coef_df = pd.DataFrame(feature_importance, index=X.columns, columns=['Coefficient'])

# Sort the coefficients in descending order (optional)
coef_df = coef_df.sort_values(by='Coefficient', ascending=False)

# Plotting
plt.figure(figsize=(10, 6))
plt.barh(coef_df.index, coef_df['Coefficient'], color='steelblue')
plt.xlabel('Coefficient Value')
plt.title('Feature Importance (Logistic Regression)')
plt.grid(True, axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()


# In[89]:


from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
report = classification_report(y_test, y_pred)
print(report)


# In[90]:


feature_importance = model.feature_importances_

# Create a DataFrame for easier handling
feature_importance_df = pd.DataFrame(feature_importance, index=X_train.columns, columns=['Importance'])

# Sort the coefficients for better visualization
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=True)

# Plotting
plt.figure(figsize=(8, 4))
plt.barh(feature_importance_df.index, feature_importance_df['Importance'], color='steelblue')
plt.xlabel('Importance Value')
plt.grid(True, axis='x', linestyle='--', alpha=0.7)
plt.title('Feature Importance in Random Forest Classifier')
plt.tight_layout()
plt.show()


# In[91]:


from sklearn.ensemble import GradientBoostingClassifier

model = GradientBoostingClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
report = classification_report(y_test, y_pred)
print(report)


# In[95]:


feature_importance = model.feature_importances_

# Create a DataFrame for easier handling
feature_importance_df = pd.DataFrame(feature_importance, index=X_train.columns, columns=['Importance'])

# Sort the coefficients for better visualization
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=True)

# Plotting
plt.figure(figsize=(8, 4))
plt.barh(feature_importance_df.index, feature_importance_df['Importance'], color='steelblue')
plt.xlabel('Importance Value')
plt.grid(True, axis='x', linestyle='--', alpha=0.7)
plt.title('Feature Importance in Gradient Boosting Classifier')
plt.tight_layout()
plt.show()


# In[97]:


from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from scipy.stats import uniform, randint
import time

# Set up cross-validation strategy
cv_strategy = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

# Parameter distributions
lr_param_dist = {
    'classifier__C': uniform(0.001, 1),
    'classifier__solver': ['liblinear'],
    'classifier__max_iter': randint(10, 100)
}

rf_param_dist = {
    'classifier__n_estimators': randint(50, 100),
    'classifier__max_depth': randint(3, 20),
    'classifier__min_samples_split': randint(2, 10),
    'classifier__min_samples_leaf': randint(1, 10),
    'classifier__max_features': ['sqrt', 'log2', None]
}

gb_param_dist = {
    'classifier__n_estimators': randint(50, 100),
    'classifier__learning_rate': uniform(0.01, 0.5),
    'classifier__max_depth': randint(3, 10),
    'classifier__min_samples_split': randint(2, 10),
    'classifier__min_samples_leaf': randint(1, 10),
    'classifier__max_features': ['sqrt', 'log2', None]
}

# Create pipelines
def model_pipeline(model, scale=False):
    steps = []
    if scale:
        steps.append(('scaler', StandardScaler()))
    steps.append(('classifier', model))
    return Pipeline(steps)

pipelines = {
    'Logistic Regression': (model_pipeline(LogisticRegression(random_state=42), scale=True), lr_param_dist),
    'Random Forest': (model_pipeline(RandomForestClassifier(random_state=42)), rf_param_dist),
    'Gradient Boosting': (model_pipeline(GradientBoostingClassifier(random_state=42)), gb_param_dist)
}

# Run RandomizedSearchCV
n_iter_search = 5
best_models, scores, train_times = {}, {}, {}

for name, (pipe, params) in pipelines.items():
    print(f"\n{'='*20} {name} {'='*20}")
    start = time.time()

    random_search = RandomizedSearchCV(
        pipe,
        param_distributions=params,
        n_iter=n_iter_search,
        cv=cv_strategy,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1,
        random_state=42
    )

    random_search.fit(X_train, y_train)

    duration = time.time() - start
    train_times[name] = duration
    best_models[name] = random_search.best_estimator_

    y_pred = random_search.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    scores[name] = acc

    print(f"\nBest Parameters: {random_search.best_params_}")
    print(f"Training Time: {duration:.2f} seconds")
    print(f"Test Accuracy: {acc:.4f}")
    print(f"ROC AUC Score: {roc_auc_score(y_test, y_pred):.4f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))


# In[99]:


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split

# Split the dataset
X = df_train.drop('satisfaction', axis=1)
y = df_train['satisfaction']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Initialize models
models = {
    'Logistic Regression': LogisticRegression(C=0.1, solver='liblinear', max_iter=500, random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42)
}

# Store results
results = {}

# Evaluate each model
for model_name, model in models.items():
    print(f"\nTraining {model_name}...")
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    
    # Store classification report and accuracy
    results[model_name] = {
        'accuracy': accuracy,
        'classification_report': classification_report(y_test, y_pred)
    }
    
    # Print classification report
    print(f"\n{model_name} Classification Report:")
    print(results[model_name]['classification_report'])
    print(f"Accuracy: {accuracy:.4f}")

# Find the best model
best_model_name = max(results, key=lambda x: results[x]['accuracy'])
best_model = models[best_model_name]
best_accuracy = results[best_model_name]['accuracy']

print(f"\nBest Model: {best_model_name}")
print(f"Best Accuracy: {best_accuracy:.4f}")


# __ML flow__

# In[104]:


import mlflow
import mlflow.sklearn

#models = [
    (
        "Logistic Regression", 
        {"C": uniform(0.001, 1), "solver": 'liblinear', "max_iter" : randint(10, 100)},
        LogisticRegression(), 
        (X_train, y_train),
        (X_test, y_test)
    ),
    (
        "Random Forest Classifier", 
        {"max_depth": randint(3, 20), "max_features": ['sqrt', 'log2', None], "min_samples_leaf":randint(1, 10), "min_samples_split":randint(2, 10), "n_estimators":randint(50, 100)},
        RandomForestClassifier(), 
        (X_train, y_train),
        (X_test, y_test)
    ),
    (
        "Gradient Boosting Classifier",
        {"learning_rate":uniform(0.01, 0.5), "max_depth" : randint(3, 10), "min_samples_leaf" : randint(1, 10), "min_samples_split" :randint(2, 10), "n_estimators" :randint(50, 100)},
        GradientBoostingClassifier(), 
        (X_train, y_train),
        (X_test, y_test)
    ),
]
# In[121]:


import numpy as np
from scipy.stats import uniform, randint
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score
import random

# Function to sample actual values from distributions
def sample_params(param_dist, random_state=42):
    np.random.seed(random_state)
    sampled = {}
    for key, val in param_dist.items():
        if hasattr(val, "rvs"):  # it's a distribution
            sampled[key] = val.rvs(random_state=random_state)
        elif isinstance(val, list):
            sampled[key] = random.choice(val)
        else:
            sampled[key] = val
    return sampled

# Define model configs
models = [
    (
        "Logistic Regression", 
        {"C": uniform(0.001, 1), "solver": ['liblinear'], "max_iter": randint(10, 100)},
        LogisticRegression(),
        (X_train, y_train),
        (X_test, y_test)
    ),
    (
        "Random Forest Classifier", 
        {"max_depth": randint(3, 20), "max_features": ['sqrt', 'log2', None], 
         "min_samples_leaf": randint(1, 10), "min_samples_split": randint(2, 10), 
         "n_estimators": randint(50, 100)},
        RandomForestClassifier(),
        (X_train, y_train),
        (X_test, y_test)
    ),
    (
        "Gradient Boosting Classifier",
        {"learning_rate": uniform(0.01, 0.5), "max_depth": randint(3, 10),
         "min_samples_leaf": randint(1, 10), "min_samples_split": randint(2, 10),
         "n_estimators": randint(50, 100)},
        GradientBoostingClassifier(),
        (X_train, y_train),
        (X_test, y_test)
    )
]

# Run and evaluate each model
results = {}

for name, param_dist, model, train_set, test_set in models:
    print(f"\n📌 Training {name}...")

    # Sample parameters
    sampled_params = sample_params(param_dist, random_state=42)

    # Set and fit the model
    model.set_params(**sampled_params)
    X_train, y_train = train_set
    X_test, y_test = test_set
    model.fit(X_train, y_train)
    
    # Predictions and evaluation
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    results[name] = {
        "params": sampled_params,
        "accuracy": acc,
        "report": report
    }

    print(f"✅ Accuracy: {acc:.4f}")
    print(f"Sampled Parameters: {sampled_params}")
    print(classification_report(y_test, y_pred))


# In[127]:


import mlflow
import mlflow.sklearn
from sklearn.metrics import classification_report, accuracy_score
from scipy.stats import uniform, randint
import random
import numpy as np

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("Customer Satisfaction Detection")

# Function to sample actual values
def sample_params(param_dist, random_state=42):
    np.random.seed(random_state)
    sampled = {}
    for key, val in param_dist.items():
        if hasattr(val, "rvs"):  # distribution
            sampled[key] = val.rvs(random_state=random_state)
        elif isinstance(val, list):
            sampled[key] = random.choice(val)
        else:
            sampled[key] = val
    return sampled

# Define models
models = [
    (
        "Logistic Regression", 
        {"C": uniform(0.001, 1), "solver": ['liblinear'], "max_iter": randint(10, 100)},
        LogisticRegression(),
        (X_train, y_train),
        (X_test, y_test)
    ),
    (
        "Random Forest Classifier", 
        {"max_depth": randint(3, 20), "max_features": ['sqrt', 'log2', None], 
         "min_samples_leaf": randint(1, 10), "min_samples_split": randint(2, 10), 
         "n_estimators": randint(50, 100)},
        RandomForestClassifier(),
        (X_train, y_train),
        (X_test, y_test)
    ),
    (
        "Gradient Boosting Classifier",
        {"learning_rate": uniform(0.01, 0.5), "max_depth": randint(3, 10),
         "min_samples_leaf": randint(1, 10), "min_samples_split": randint(2, 10),
         "n_estimators": randint(50, 100)},
        GradientBoostingClassifier(),
        (X_train, y_train),
        (X_test, y_test)
    )
]

# Train and log
for name, param_dist, model, train_set, test_set in models:
    sampled_params = sample_params(param_dist)
    
    model.set_params(**sampled_params)
    X_train, y_train = train_set
    X_test, y_test = test_set
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    with mlflow.start_run(run_name=name):
        # ✅ Log real sampled parameters
        mlflow.log_params(sampled_params)

        # ✅ Log metrics
        mlflow.log_metrics({
            'accuracy': acc,
            'recall_class_1': report['1']['recall'],
            'recall_class_0': report['0']['recall'],
            'f1_score_macro': report['macro avg']['f1-score']
        })

        # ✅ Log model
        mlflow.sklearn.log_model(model, "model")

        print(f"✅ Logged {name} to MLflow")

reports = []

for model_name, params, model, train_set, test_set in models:
    X_train = train_set[0]
    y_train = train_set[1]
    X_test = test_set[0]
    y_test = test_set[1]
    
    model.set_params(**params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    reports.append(report)mlflow.set_experiment("Customer Satisfaction Detection")
mlflow.set_tracking_uri("http://localhost:5000")

for i, element in enumerate(models):
    model_name = element[0]
    params = element[1]
    model = element[2]
    report = reports[i]
    
    with mlflow.start_run(run_name=model_name):        
        mlflow.log_params(params)
        mlflow.log_metrics({
            'accuracy': report['accuracy'],
            'recall_class_1': report['1']['recall'],
            'recall_class_0': report['0']['recall'],
            'f1_score_macro': report['macro avg']['f1-score']
        })  
        
        if "XGB" in model_name:
            mlflow.xgboost.log_model(model, "model")
        else:
            mlflow.sklearn.log_model(model, "model")  import mlflow
import mlflow.sklearn
from sklearn.metrics import accuracy_score, f1_score

# Best model from previous evaluation
final_model = best_model  # Use the best model identified above
final_model.fit(X_train, y_train)  # Ensure it is fitted

# Predict using the final model
y_pred = final_model.predict(X_test)

# Start MLflow run
with mlflow.start_run(run_name="Final_Model_Registration") as run:
    
    # Log model parameters
    if hasattr(final_model, 'get_params'):
        params = final_model.get_params()
        for key, value in params.items():
            mlflow.log_param(key, value)
    else:
        print("Model does not support get_params()")

    # Log evaluation metrics
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("f1_score", f1)

    # Log the model itself
    mlflow.sklearn.log_model(final_model, "model")

    # Print run information
    print(f"Run ID: {run.info.run_id}")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1 Score: {f1:.4f}")

# In[ ]:


import mlflow
import mlflow.sklearn
from sklearn.metrics import accuracy_score, f1_score

# Best model from previous evaluation
final_model = best_model
final_model.fit(X_train, y_train)

# Predict using the final model
y_pred = final_model.predict(X_test)

with mlflow.start_run(run_name="Final_Model_Registration") as run:
    
    # Log parameters
    if hasattr(final_model, 'get_params'):
        params = final_model.get_params()
        for key, value in params.items():
            mlflow.log_param(key, value)

    # Log evaluation metrics
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("f1_score", f1)

    # Log model
    mlflow.sklearn.log_model(final_model, "model")

    # Register the model
    model_uri = f"runs:/{run.info.run_id}/model"
    mlflow.register_model(model_uri=model_uri, name="Customer_Satisfaction_Model")

    print(f"Run ID: {run.info.run_id}")
    print(f"Accuracy: {acc:.4f}")


# In[138]:


import mlflow

# Set the correct tracking URI (if not set already)
mlflow.set_tracking_uri("http://localhost:5000")  # or your appropriate server URI

# Get a list of all runs in the experiment
client = mlflow.tracking.MlflowClient()
experiment_id = "284823388404894191"  # Use the correct experiment ID
runs = client.search_runs(experiment_id)

# Print all the runs and their IDs
for run in runs:
    print(f"Run ID: {run.info.run_id}, Status: {run.info.status}")


# In[146]:


import mlflow

# Set the correct tracking URI (if not set already)
mlflow.set_tracking_uri("http://localhost:5000")  # or your appropriate server URI

# Specify the run ID and load the model
run_id = "9f1b8869529a48fda5feb3c0ccb4cd59"  # Replace with the appropriate run ID
model_uri = f"runs:/{run_id}/model"

# Load the model (replace with appropriate flavor, e.g., mlflow.sklearn for sklearn models)
flight_model = mlflow.sklearn.load_model(model_uri)

# Print success message
print("Model loaded successfully.")


# In[148]:


import mlflow

run_id = "9f1b8869529a48fda5feb3c0ccb4cd59"  # Replace with your run ID
model_uri = f"runs:/{run_id}/model"

# Get the model's metadata
model = mlflow.pyfunc.load_model(model_uri)
print(model)


# In[102]:


import mlflow
from mlflow.tracking import MlflowClient
from mlflow.sklearn import load_model

# Step 1: Connect to MLflow tracking server
mlflow.set_tracking_uri("http://127.0.0.1:5000")

# Step 2: Define your Customer Satisfaction experiment ID
experiment_id = "151365011729759780"  # Your actual experiment ID

# Step 3: Search for runs and sort by f1_score_macro
client = MlflowClient()

runs = client.search_runs(
    experiment_ids=[experiment_id],
    order_by=["metrics.f1_score_macro DESC"],  # Sort by F1
    max_results=1  # Top 1
)

# Step 4: Get best run and model details
best_run = runs[0]
best_run_id = best_run.info.run_id
best_run_name = best_run.data.tags.get("mlflow.runName")
f1_score = best_run.data.metrics["f1_score_macro"]
accuracy = best_run.data.metrics["accuracy"]

print(f"✅ Best Model Found:")
print(f"Run ID: {best_run_id}")
print(f"Model Name: {best_run_name}")
print(f"F1 Score (Macro): {f1_score}")
print(f"Accuracy: {accuracy}")

# Step 5: Load the best model
model_uri = f"runs:/{best_run_id}/model"
best_model = load_model(model_uri)




# In[ ]:




