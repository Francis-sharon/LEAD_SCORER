import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression

# Load the dataset
path = r"C:\Users\91735\Desktop\main.py"
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

# Function to make lead prediction
def make_lead_prediction(input_values):
    # Ask for user input for different features
    feature_values = {}
    for i, column in enumerate(X.columns):
        value = input_values[i]
        print(f"Using input value for {column} ({input_labels[i]}): {value}")
        feature_values[column] = value

    for column in string_columns:
        feature_values[column] = label_encoders[column].transform([feature_values[column]])[0]

    for column in numeric_columns:
        feature_values[column] = float(feature_values[column])

    user_input = pd.DataFrame(feature_values, index=[0])

    # Make predictions on the user input
    user_pred = model.predict(user_input)

    # Convert the predictions to a DataFrame or use as needed
    #user_predictions_df = pd.DataFrame({'Converted': user_pred})
    if user_pred == 0:
        print("Lead is of less Focus")
        print("MODEL SCORE: AVERAGE LEAD")
    else:
        print("Lead is of Higher Focus")
        print("MODEL SCORE: HOT LEAD")

if __name__ == '__main__':
    print("Lead Prediction Program Started")