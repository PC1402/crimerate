import pickle

# Load the model from the pickle file
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

# Check if the model is loaded correctly
print(model)

# If you want to see the model's details (like hyperparameters, if available)
# You can print its type and check the model
print(type(model))
