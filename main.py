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

# Calculate the confusion matrix
confusion_matrix = confusion_matrix(y_test, y_pred)

acc = accuracy_score(y_test, y_pred)
print("\nOVERALL ACCURACY SCORE: ", acc)

sns.heatmap(confusion_matrix/np.sum(confusion_matrix), annot=True,
            fmt='.2%', cmap='Blues').set(title='Confusion matrix: Logistic Regression')

r = classification_report(y_test, y_pred)
print("\n CLASSIFICATION REPORT: \n", r)

# Ask for user input for different features
feature_values = {}
for column in X.columns:
    if column == "Unnamed: 0":
        feature_values[column] = 1
    else:
        value = input(f"Enter the value for {column}: ")
        print(f"{column}: {value}")  # print user input for debugging
        feature_values[column] = value
# Preprocess user input (e.g., transform using LabelEncoder)
for column in string_columns:
    feature_values[column] = label_encoders[column].transform([feature_values[column]])[0]

# Convert numeric features to appropriate type
for column in numeric_columns:
    feature_values[column] = float(feature_values[column])

# Create a new DataFrame with the user input
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
    print("MODEL SCORE: HOT lEAD")

# Print the predictions or use as needed


# ... (previous code)

# Ask for user input for different features

# ... (rest of the code)
