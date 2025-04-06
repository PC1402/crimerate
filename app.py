from flask import Flask, request, jsonify, send_from_directory
import os
import pickle
import pandas as pd
from datetime import datetime
import requests

app = Flask(__name__)

MODEL_URL = "https://drive.google.com/uc?export=download&id=1eBoBqYIlnA_7I5kJ6pCJo-V6j36UU47Q"
MODEL_PATH = "model.pkl"
ENCODER_PATH = "label_encoders.pkl"

# ✅ Download model from Google Drive if not found
def download_file(url, destination):
    if not os.path.exists(destination):
        print(f"📦 Downloading {destination}...")
        response = requests.get(url)
        with open(destination, 'wb') as f:
            f.write(response.content)
        print(f"✅ Downloaded {destination}")

# 📥 Check and download model + encoder
download_file(MODEL_URL, MODEL_PATH)

# 🔁 Load model and encoders
with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)

with open(ENCODER_PATH, 'rb') as le_file:
    label_encoders = pickle.load(le_file)

# ✅ Get training feature names dynamically
training_columns = list(model.estimators_[0].feature_names_in_)
print("Model expects these columns:", training_columns)

# 🌐 Serve pages and assets
@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/dashboard.html')
def dashboard():
    return send_from_directory('.', 'dashboard.html')

@app.route('/<path:filename>')
def static_files(filename):
    return send_from_directory('.', filename)

# 🔮 Prediction Endpoint
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        city = data.get('City')
        date_str = data.get('Date of Occurrence')
        time_str = data.get('Time of Occurrence')

        date_obj = pd.to_datetime(date_str, format='%Y-%m-%d')
        time_obj = pd.to_datetime(time_str, format='%H:%M')

        input_df = pd.DataFrame([{col: 0 for col in training_columns}])
        input_df.at[0, 'Day of Year'] = date_obj.dayofyear
        input_df.at[0, 'Month'] = date_obj.month
        input_df.at[0, 'Day of Week'] = date_obj.weekday()
        input_df.at[0, 'Hour'] = time_obj.hour
        input_df.at[0, 'Minute'] = time_obj.minute

        city_col = f'City_{city}'
        if city_col in input_df.columns:
            input_df.at[0, city_col] = 1
        else:
            valid_cities = [col.replace('City_', '') for col in training_columns if col.startswith('City_')]
            return jsonify({"error": f"City '{city}' not recognized. Valid cities: {valid_cities}"}), 400

        prediction = model.predict(input_df)[0]

        decoded = {}
        output_keys = ['Crime Description', 'Victim Age', 'Victim Gender', 'Crime Domain']
        for i, key in enumerate(output_keys):
            if key in label_encoders:
                decoded_val = label_encoders[key].inverse_transform([prediction[i]])[0]
            else:
                decoded_val = prediction[i]
            decoded[key] = str(decoded_val) if isinstance(decoded_val, str) else int(decoded_val)

        return jsonify(decoded)

    except Exception as e:
        print(f"❌ Error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
