import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.multioutput import MultiOutputClassifier
import pickle

# Load the dataset
df = pd.read_csv("crime_data.csv")

# Convert 'Date of Occurrence' to datetime
df['Date of Occurrence'] = pd.to_datetime(df['Date of Occurrence'], format='%Y-%m-%d')

# Extract date-related features from 'Date of Occurrence'
df['Day of Year'] = df['Date of Occurrence'].dt.dayofyear
df['Month'] = df['Date of Occurrence'].dt.month
df['Day of Week'] = df['Date of Occurrence'].dt.weekday  # Ensure this is numeric

# Convert 'Time of Occurrence' to datetime and extract hour and minute
df['Time of Occurrence'] = pd.to_datetime(df['Time of Occurrence'], format='%H:%M')
df['Hour'] = df['Time of Occurrence'].dt.hour
df['Minute'] = df['Time of Occurrence'].dt.minute

# Drop original 'Date of Occurrence' and 'Time of Occurrence'
df.drop(columns=['Date of Occurrence', 'Time of Occurrence'], inplace=True)

# One-hot encode the 'City' column
df = pd.get_dummies(df, columns=['City'], drop_first=True)

# Label encode categorical columns
label_encoders = {}
for column in ['Crime Description', 'Victim Gender', 'Crime Domain']:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

# Ensure 'Day of Week' is numeric (it should be after the extraction above, but let's ensure this)
df['Day of Week'] = pd.to_numeric(df['Day of Week'], errors='coerce')

# Separate features (X) and target (y)
X = df.drop(['Crime Description', 'Victim Age', 'Victim Gender', 'Crime Domain'], axis=1)
y = df[['Crime Description', 'Victim Age', 'Victim Gender', 'Crime Domain']]

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the RandomForest model and wrap it with MultiOutputClassifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
multi_output_model = MultiOutputClassifier(rf_model, n_jobs=-1)

# Train the model
multi_output_model.fit(X_train, y_train)

# Evaluate the model on test data
accuracy = multi_output_model.score(X_test, y_test)
print(f"Model accuracy: {accuracy * 100:.2f}%")

# Save the trained model to 'model.pkl'
with open('model.pkl', 'wb') as model_file:
    pickle.dump(multi_output_model, model_file)

# Save the label encoders to 'label_encoders.pkl'
with open('label_encoders.pkl', 'wb') as le_file:
    pickle.dump(label_encoders, le_file)

print("Model and label encoders have been saved.")
