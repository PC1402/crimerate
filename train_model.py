import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.multioutput import MultiOutputClassifier
import pickle

# Load dataset
df = pd.read_csv("crime_data.csv")

# Parse and extract datetime features
df['Date of Occurrence'] = pd.to_datetime(df['Date of Occurrence'], format='%Y-%m-%d')
df['Time of Occurrence'] = pd.to_datetime(df['Time of Occurrence'], format='%H:%M')

df['Day of Year'] = df['Date of Occurrence'].dt.dayofyear
df['Month'] = df['Date of Occurrence'].dt.month
df['Day of Week'] = df['Date of Occurrence'].dt.weekday
df['Hour'] = df['Time of Occurrence'].dt.hour
df['Minute'] = df['Time of Occurrence'].dt.minute

df.drop(columns=['Date of Occurrence', 'Time of Occurrence'], inplace=True)

# One-hot encode City
df = pd.get_dummies(df, columns=['City'], drop_first=True)

# Label encode target columns
label_encoders = {}
for column in ['Crime Description', 'Victim Gender', 'Crime Domain']:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

# Prepare features and targets
X = df.drop(['Crime Description', 'Victim Age', 'Victim Gender', 'Crime Domain'], axis=1)
y = df[['Crime Description', 'Victim Age', 'Victim Gender', 'Crime Domain']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Try different tree counts to balance size vs. accuracy
for n in [10, 25, 50]:
    print(f"\n🌲 Training RandomForest with {n} trees...")
    rf = RandomForestClassifier(n_estimators=n, random_state=42)
    model = MultiOutputClassifier(rf)
    model.fit(X_train, y_train)
    acc = model.score(X_test, y_test)
    print(f"✅ Accuracy with {n} trees: {acc:.2%}")

# Save model with chosen tree count (here: 25)
final_rf = RandomForestClassifier(n_estimators=25, random_state=42)
final_model = MultiOutputClassifier(final_rf)
final_model.fit(X_train, y_train)

with open('model.pkl', 'wb') as model_file:
    pickle.dump(final_model, model_file, protocol=pickle.HIGHEST_PROTOCOL)

with open('label_encoders.pkl', 'wb') as le_file:
    pickle.dump(label_encoders, le_file, protocol=pickle.HIGHEST_PROTOCOL)

print("\n💾 Saved model.pkl and label_encoders.pkl with 25 trees.")
