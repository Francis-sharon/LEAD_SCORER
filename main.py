import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import seaborn as sns
import numpy as np

# Load the dataset
path = r"C:\Users\JR.SHARON\Desktop\leads-kaggle\xy.csv"
df = pd.read_csv(path)

# Missing values fill
string_columns = df.select_dtypes(include=['object']).columns
numeric_columns = df.select_dtypes(include=['float']).columns

df[string_columns] = df[string_columns].fillna('Unknown')
df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())

# Encode string columns using ordinal encoding
label_encoders = {}
for column in string_columns:
    label_encoders[column] = LabelEncoder()
    df[column] = label_encoders[column].fit_transform(df[column])

m_c = df.isnull().sum()
empty = m_c[m_c == len(df)].index
df = df.drop(empty, axis=1)

# Data Split
y = df['Converted']
X = df.drop(['Converted'], axis=1)  # Exclude 'Lead Number'

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the Logistic Regression classifier
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Ask for user input for different features
feature_values = {}
for column in X.columns:
    if column == "Unnamed: 0":
        feature_values[column] = 1
    elif column == "Lead Origin":
        value = input(f"Enter the value for {column} (API/Website): ")
        print(f"{column}: {value}")  
        feature_values[column] = value
    elif column == "Lead Source":
        value = input(f"Enter the value for {column} (Reference/Direct traffic): ")
        print(f"{column}: {value}")  
        feature_values[column] = value
    elif column == "Do Not Call":
        value = input(f"Enter the value for {column} (yes/no): ")
        print(f"{column}: {value}")  
        feature_values[column] = value
    elif column == "Do Not Email":
        value = input(f"Enter the value for {column} (yes/no): ")
        print(f"{column}: {value}")  
        feature_values[column] = value
    elif column == "TotalVisits":
        value = input(f"Enter the value for {column} (in numbers): ")
        print(f"{column}: {value}")  
        feature_values[column] = value
    elif column == "Total Time Spent on Website":
        value = input(f"Enter the value for {column} (in hours): ")
        print(f"{column}: {value}")  
        feature_values[column] = value
    elif column == "Last Activity":
        value = input(f"Enter the value for {column} (page visited/email opened): ")
        print(f"{column}: {value}")  
        feature_values[column] = value
    elif column == "Country":
        value = input(f"Enter the value for {column}: ")
        print(f"{column}: {value}")  
        feature_values[column] = value
    elif column == "employment":
        value = input(f"Enter the value for {column} (employed/unemployed/Student): ")
        print(f"{column}: {value}")  
        feature_values[column] = value
    elif column == "city":
        value = input(f"Enter the value for {column}: ")
        print(f"{column}: {value}")  
        feature_values[column] = value
for column in string_columns:
    feature_values[column] = label_encoders[column].transform([feature_values[column]])[0]

for column in numeric_columns:
    feature_values[column] = float(feature_values[column])

user_input = pd.DataFrame(feature_values, index=[0])

# Make predictions on the user input
user_pred = model.predict(user_input)
print(user_predictions_df)

# Convert the predictions to a DataFrame or use as needed
#user_predictions_df = pd.DataFrame({'Converted': user_pred})
if user_pred == 0:
    print("Lead is of less Focus")
    print("MODEL SCORE: AVERAGE LEAD")
else:
    print("Lead is of Higher Focus")
    print("MODEL SCORE: HOT LEAD")
