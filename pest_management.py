import pandas as pd
import pickle

# Load the dataset
dataset = pd.read_csv("crop_recommendation.csv")

# Select key features
selected_features = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]

# Extract the selected features from the dataset
X = dataset[selected_features]

# Load the trained Gaussian Naive Bayes model
with open("crop_recommendation.pkl", "rb") as f:
    model = pickle.load(f)

# Create new features based on aggregation
X["temperature_mean"] = X["temperature"].mean()
X["humidity_std"] = X["humidity"].std()
X["rainfall_max"] = X["rainfall"].max()

# Create new features based on interaction terms
X["temperature_humidity_interaction"] = X["temperature"] * X["humidity"]

# Create new features based on time information
X["month"] = pd.to_datetime(dataset["date"]).dt.month
X["season"] = pd.cut(
    X["month"], bins=[0, 3, 6, 9, 12], labels=["Winter", "Spring", "Summer", "Autumn"]
)

# Define pest outbreak thresholds for rainfall and humidity
rainfall_threshold = (
    100  # Example threshold for rainfall (you can adjust it according to your needs)
)
humidity_threshold = (
    70  # Example threshold for humidity (you can adjust it according to your needs)
)

# Predict likelihood of pest outbreaks for new data
new_data = pd.DataFrame(
    {
        "N": [25],
        "P": [50],
        "K": [40],
        "temperature": [28],
        "humidity": [75],
        "ph": [6.5],
        "rainfall": [100],
        "temperature_mean": [X["temperature"].mean()],
        "humidity_std": [X["humidity"].std()],
        "rainfall_max": [X["rainfall"].max()],
        "temperature_humidity_interaction": [28 * 75],
        "month": [6],
        "season": ["Summer"],
    }
)

predicted_outcome = model.predict(new_data)

# Check for pest outbreak based on rainfall and humidity
pest_outbreak = False

if (
    new_data["rainfall"].values[0] > rainfall_threshold
    and new_data["humidity"].values[0] > humidity_threshold
):
    pest_outbreak = True

# Print the predicted outcome and pest outbreak possibility
print("Predicted outcome:", predicted_outcome)
print("Pest outbreak possibility:", pest_outbreak)
