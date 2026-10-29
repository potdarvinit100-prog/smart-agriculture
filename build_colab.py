"""Build the simple Google Colab project for Smart Agriculture Monitoring.

Outputs (in ~/workspace/your_files/smart-agriculture-colab/):
  - Smart_Agriculture_Colab.ipynb  (runs as-is in Colab, no uploads needed)
  - agriculture_sensor_data.csv    (the same 300-row dataset the notebook makes)

Design (kept deliberately simple, mirroring the attendance sample):
  - 300 sensor readings, 5 input columns + 1 label.
  - Label follows the project's own irrigation rule: irrigation is needed
    when soil moisture is below the crop's ideal minimum
    (wheat 40, rice 60, tomato 50, onion 45).
  - 80/20 train/test split, RandomForestClassifier, accuracy + report.
"""
import nbformat as nbf
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path("/home/hatch/workspace/your_files/smart-agriculture-colab")
OUT.mkdir(parents=True, exist_ok=True)

CROPS_MIN_MOISTURE = {"wheat": 40, "rice": 60, "tomato": 50, "onion": 45}
CROPS = list(CROPS_MIN_MOISTURE)


def make_dataset(seed=42, n=300):
    rng = np.random.default_rng(seed)
    crop = rng.choice(CROPS, size=n)
    moisture = rng.uniform(20, 90, size=n).round(1)
    temp = rng.uniform(15, 40, size=n).round(1)
    humidity = rng.uniform(40, 95, size=n).round(1)
    light = rng.uniform(100, 1100, size=n).round(0).astype(int)
    need = np.array(
        [1 if m < CROPS_MIN_MOISTURE[c] else 0 for m, c in zip(moisture, crop)],
        dtype=int,
    )
    return pd.DataFrame(
        {
            "crop": crop,
            "soil_moisture": moisture,
            "temperature": temp,
            "humidity": humidity,
            "light": light,
            "irrigation_needed": need,
        }
    )


df = make_dataset()
csv_path = OUT / "agriculture_sensor_data.csv"
df.to_csv(csv_path, index=False)
print("CSV written:", csv_path, df.shape)
print(df["irrigation_needed"].value_counts().to_dict())

# ---------------- Notebook ----------------
nb = nbf.v4.new_notebook()
nb.metadata = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.10"},
}

md_title = """# Smart Agriculture Monitoring — Simple ML Project (Google Colab)

**Aim:** Predict whether a crop needs irrigation, from sensor readings.

**How to run:** Open this notebook in Google Colab and press **Runtime → Run all**.
No files to upload and nothing to install — pandas and scikit-learn are
already available in Colab. The first code cell creates the dataset
(`agriculture_sensor_data.csv`, 300 readings) and the same CSV file is
also included with this project, so the data you submit matches the
notebook exactly.

**How the label is made:** irrigation is needed (1) when soil moisture is
below the crop's ideal minimum — wheat 40%, rice 60%, tomato 50%, onion
45%. Otherwise it is not needed (0). The model learns this rule from the
data; it is a demonstration, not advice tested on a real field."""

code_gen = '''# Cell 1: Create the dataset and save it as a CSV file
import numpy as np
import pandas as pd

# Ideal minimum soil moisture for each crop (same rule as the project)
CROPS_MIN_MOISTURE = {"wheat": 40, "rice": 60, "tomato": 50, "onion": 45}
CROPS = list(CROPS_MIN_MOISTURE)

np.random.seed(42)  # fixed seed, so the result is the same every run
n = 300

crop = np.random.choice(CROPS, size=n)
soil_moisture = np.round(np.random.uniform(20, 90, size=n), 1)
temperature = np.round(np.random.uniform(15, 40, size=n), 1)
humidity = np.round(np.random.uniform(40, 95, size=n), 1)
light = np.random.randint(100, 1100, size=n)

# Label: 1 = irrigation needed, 0 = not needed
irrigation_needed = [1 if m < CROPS_MIN_MOISTURE[c] else 0
                     for m, c in zip(soil_moisture, crop)]

data = pd.DataFrame({
    "crop": crop,
    "soil_moisture": soil_moisture,
    "temperature": temperature,
    "humidity": humidity,
    "light": light,
    "irrigation_needed": irrigation_needed,
})

data.to_csv("agriculture_sensor_data.csv", index=False)
print("Dataset shape:", data.shape)
print(data.head())
print(data["irrigation_needed"].value_counts())'''

md_train = """## Training and testing

The data is split 80% training / 20% testing. Crop names are converted to
numbers with one-hot encoding, then a Random Forest model (100 trees) is
trained and tested."""

code_train = '''# Cell 2: Split into training and testing data, train, and test
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

data = pd.read_csv("agriculture_sensor_data.csv")

X = pd.get_dummies(data.drop("irrigation_needed", axis=1), columns=["crop"])
y = data["irrigation_needed"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42)

print("Training data:", X_train.shape)
print("Testing data:", X_test.shape)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Test accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))'''

md_try = """## Try it on new readings

Three new sensor readings are checked with the trained model. The plain-word
answer is printed next to each one."""

code_try = '''# Cell 3: Predict irrigation for new sensor readings
new_readings = pd.DataFrame({
    "crop": ["wheat", "tomato", "rice"],
    "soil_moisture": [28, 62, 55],
    "temperature": [34, 27, 30],
    "humidity": [55, 70, 80],
    "light": [950, 800, 600],
})

new_X = pd.get_dummies(new_readings, columns=["crop"])
new_X = new_X.reindex(columns=X.columns, fill_value=0)

for reading, pred in zip(new_readings.to_dict("records"), model.predict(new_X)):
    answer = "YES - Irrigate now" if pred == 1 else "NO - Soil is good"
    print(reading, "->", answer)'''

nb.cells = [
    nbf.v4.new_markdown_cell(md_title),
    nbf.v4.new_code_cell(code_gen),
    nbf.v4.new_markdown_cell(md_train),
    nbf.v4.new_code_cell(code_train),
    nbf.v4.new_markdown_cell(md_try),
    nbf.v4.new_code_cell(code_try),
]
nbf.validate(nb)
ipynb_path = OUT / "Smart_Agriculture_Colab.ipynb"
nbf.write(nb, str(ipynb_path))
print("Notebook written:", ipynb_path)
