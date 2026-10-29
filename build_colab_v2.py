"""Build v2 of the Smart Agriculture Colab notebook, in the style of the
user's finance snippet: choose/upload a CSV file, preprocess, split,
train, evaluate, save the model as a .pkl file.

Difference from the finance snippet, kept deliberately: irrigation is a
yes/no target, so this uses a RandomForestClassifier + accuracy instead
of a RandomForestRegressor + MAE. Everything else follows the same shape.

Outputs (in ~/workspace/your_files/smart-agriculture-colab/):
  - Smart_Agriculture_Colab.ipynb (overwrites v1)
  - agriculture_sensor_data.csv   (unchanged, the file to upload)
  - agriculture_model.pkl         (produced by the verification run)
"""
import nbformat as nbf
from pathlib import Path

OUT = Path("/home/hatch/workspace/your_files/smart-agriculture-colab")

nb = nbf.v4.new_notebook()
nb.metadata = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.10"},
}

md_title = """# Smart Agriculture Monitoring — Irrigation Prediction (Google Colab)

**Aim:** Predict whether a crop needs irrigation, from sensor readings.

**How to run:** Open this notebook in Google Colab and press **Runtime → Run all**.
When the **Choose Files** button appears, pick `agriculture_sensor_data.csv`
(the CSV file included with this project). The rest runs by itself — no
libraries to install, pandas and scikit-learn are already in Colab.

**Note:** Your finance example predicted an amount (a number), so it used a
regressor and MAE. Irrigation here is a yes/no answer, so this notebook uses
a classifier and accuracy instead. The shape is otherwise the same:
choose file → preprocess → split → train → evaluate → save model."""

code_upload = '''# Cell 1: Choose the CSV file and load it
import io

import pandas as pd

df = None
try:
    from google.colab import files
    uploaded = files.upload()  # click "Choose Files", pick agriculture_sensor_data.csv
    if uploaded:
        name = next(iter(uploaded))
        df = pd.read_csv(io.BytesIO(uploaded[name]))
        print("Loaded uploaded file:", name)
except ImportError:
    pass  # not running in Colab, use the local file below

if df is None:
    df = pd.read_csv("agriculture_sensor_data.csv")
    print("Loaded local file: agriculture_sensor_data.csv")

print("Dataset shape:", df.shape)
print(df.head())'''

code_train = '''# Cell 2: Preprocess, split into training and testing sets, train, evaluate, save
import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

# One-Hot Encode the crop names
df = pd.get_dummies(df, columns=["crop"])

# Define features and target
X = df.drop("irrigation_needed", axis=1)
y = df["irrigation_needed"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Training data:", X_train.shape)
print("Testing data:", X_test.shape)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print("Test accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Save the trained model to a .pkl file
with open("agriculture_model.pkl", "wb") as file:
    pickle.dump(model, file)
print("Model saved as agriculture_model.pkl")'''

md_try = """## Try the saved model on new readings

The saved model is loaded back from the .pkl file and used to answer three
new sensor readings in plain words."""

code_try = '''# Cell 3: Load the saved model and predict for new readings
import pickle

with open("agriculture_model.pkl", "rb") as file:
    saved_model = pickle.load(file)

new_readings = pd.DataFrame({
    "crop": ["wheat", "tomato", "rice"],
    "soil_moisture": [28, 62, 55],
    "temperature": [34, 27, 30],
    "humidity": [55, 70, 80],
    "light": [950, 800, 600],
})

new_X = pd.get_dummies(new_readings, columns=["crop"])
new_X = new_X.reindex(columns=X.columns, fill_value=0)

for reading, pred in zip(new_readings.to_dict("records"), saved_model.predict(new_X)):
    answer = "YES - Irrigate now" if pred == 1 else "NO - Soil is good"
    print(reading, "->", answer)'''

nb.cells = [
    nbf.v4.new_markdown_cell(md_title),
    nbf.v4.new_code_cell(code_upload),
    nbf.v4.new_code_cell(code_train),
    nbf.v4.new_markdown_cell(md_try),
    nbf.v4.new_code_cell(code_try),
]
nbf.validate(nb)
ipynb_path = OUT / "Smart_Agriculture_Colab.ipynb"
nbf.write(nb, str(ipynb_path))
print("Notebook written:", ipynb_path)
