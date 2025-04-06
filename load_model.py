import os
import requests
import pickle

MODEL_URL = "https://drive.google.com/uc?export=download&id=1eBoBqYIlnA_7I5kJ6pCJo-V6j36UU47Q"
MODEL_PATH = "model.pkl"

def download_model():
    if not os.path.exists(MODEL_PATH):
        print("📦 Downloading model from Google Drive...")
        response = requests.get(MODEL_URL)
        with open(MODEL_PATH, 'wb') as f:
            f.write(response.content)
        print("✅ Model downloaded!")

def load_model():
    # Download if not exists
    download_model()

    # Load model from pickle file
    with open(MODEL_PATH, 'rb') as file:
        model = pickle.load(file)

    # Debug info (optional)
    print("✅ Model loaded successfully.")
    print(f"📘 Model type: {type(model)}")
    print("🧠 Model preview:")
    print(model)

    return model
