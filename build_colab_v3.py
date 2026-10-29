"""Build v3 of the Smart Agriculture Colab notebook, following the telecom
sample's structure: Choose Files upload, then Investigation 1-6, plus a
final model-save cell (kept from the finance-snippet request).

Model choice: Logistic Regression, like the telecom sample (their target
was also yes/no). It fits our binary irrigation target and makes
Investigation 6 (coefficient interpretation) possible. drop_first=True
keeps one baseline crop so the crop coefficients read cleanly against it.

Outputs (in ~/workspace/your_files/smart-agriculture-colab/):
  - Smart_Agriculture_Colab.ipynb (overwrites v2)
  - agriculture_sensor_data.csv   (unchanged, the file to upload)
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

**Note on the upload button:** If it shows "No file chosen" and says the
widget is only available in the current browser session, just **rerun that
cell** — Colab only enables the button after the cell runs. That is normal
Colab behaviour, not an error in the notebook."""

code_upload = '''# Upload: choose the CSV file
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
    print("Loaded local file: agriculture_sensor_data.csv")'''

code_inv1 = '''# Investigation 1: Understanding Dataset
print(df.columns)
print("\\nTarget Variable: irrigation_needed (1 = irrigate, 0 = no need)")
print(df["irrigation_needed"].value_counts())
print("\\nFeatures:", [c for c in df.columns if c != "irrigation_needed"])
print("\\nDataset shape:", df.shape)
print(df.head())'''

code_inv2 = '''# Investigation 2: Preparing Data for Logistic Regression
from sklearn.model_selection import train_test_split

# One-Hot Encode the crop names (first crop kept as baseline)
df_enc = pd.get_dummies(df, columns=["crop"], drop_first=True)
X = df_enc.drop("irrigation_needed", axis=1)
y = df_enc["irrigation_needed"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
print("Training Samples:", len(X_train))
print("Testing Samples:", len(X_test))
print("Number of Features:", X.shape[1])
print(X_train.head())'''

code_inv3 = '''# Investigation 3: Building and Training the Logistic Regression Model
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

print("Intercept:", model.intercept_)
print("\\nRegression Coefficients:")
for feature, coef in zip(X.columns, model.coef_[0]):
    print(f"{feature} : {coef}")'''

code_inv4 = '''# Investigation 4: Predicting Irrigation Need using the Logistic Regression Model
y_pred = model.predict(X_test)

results = pd.DataFrame(
    {"Actual Status": y_test.values, "Predicted Status": y_pred})
print(results.head(10))

# Predict for one new reading (built as a DataFrame, so feature names stay valid)
new_raw = pd.DataFrame([{
    "crop": "wheat", "soil_moisture": 28, "temperature": 34,
    "humidity": 55, "light": 950,
}])
new_X = pd.get_dummies(new_raw, columns=["crop"], drop_first=True)
new_X = new_X.reindex(columns=X.columns, fill_value=0)
print("\\nPredicted Status:",
      ["YES - Irrigate now" if p == 1 else "NO - Soil is good"
       for p in model.predict(new_X)])

results["Correct Prediction"] = (
    results["Actual Status"] == results["Predicted Status"])
print(results.head(10))'''

code_inv5 = '''# Investigation 5: Evaluating the Performance of the Logistic Regression Model
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

print("Accuracy:", accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))'''

code_inv6 = '''# Investigation 6: Interpreting the Logistic Regression Model for Decision-Making
coef_df = pd.DataFrame(
    {"Feature": X.columns, "Coefficient": model.coef_[0]})
print(coef_df)
print("Intercept:", model.intercept_[0])
ranked = coef_df.assign(
    Absolute_Coefficient=coef_df["Coefficient"].abs()
).sort_values("Absolute_Coefficient", ascending=False)
print(ranked)'''

code_save = '''# Save the trained model to a .pkl file
import pickle

with open("agriculture_model.pkl", "wb") as file:
    pickle.dump(model, file)
print("Model saved as agriculture_model.pkl")'''

cells = [
    nbf.v4.new_markdown_cell(md_title),
    nbf.v4.new_code_cell(code_upload),
    nbf.v4.new_markdown_cell("## Investigation 1: Understanding Dataset"),
    nbf.v4.new_code_cell(code_inv1),
    nbf.v4.new_markdown_cell("## Investigation 2: Preparing Data for Logistic Regression"),
    nbf.v4.new_code_cell(code_inv2),
    nbf.v4.new_markdown_cell("## Investigation 3: Building and Training the Logistic Regression Model"),
    nbf.v4.new_code_cell(code_inv3),
    nbf.v4.new_markdown_cell("## Investigation 4: Predicting Irrigation Need using the Logistic Regression Model"),
    nbf.v4.new_code_cell(code_inv4),
    nbf.v4.new_markdown_cell("## Investigation 5: Evaluating the Performance of the Logistic Regression Model"),
    nbf.v4.new_code_cell(code_inv5),
    nbf.v4.new_markdown_cell("## Investigation 6: Interpreting the Logistic Regression Model for Decision-Making"),
    nbf.v4.new_code_cell(code_inv6),
    nbf.v4.new_markdown_cell("## Save the trained model"),
    nbf.v4.new_code_cell(code_save),
]
nb.cells = cells
nbf.validate(nb)
ipynb_path = OUT / "Smart_Agriculture_Colab.ipynb"
nbf.write(nb, str(ipynb_path))
print("Notebook written:", ipynb_path)
