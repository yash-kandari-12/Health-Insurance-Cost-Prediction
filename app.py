import pandas as pd
import numpy as np

# ------------------------------------------------
# STEP 1: LOAD DATASET
# ------------------------------------------------

file_path = "datasets/insurance.csv"

df = pd.read_csv(file_path)

print("\n========== DATASET LOADED SUCCESSFULLY ==========")

# ------------------------------------------------
# STEP 2: BASIC INFORMATION
# ------------------------------------------------

print("\n1. FIRST FIVE ROWS")
print(df.head())

print("\n2. LAST FIVE ROWS")
print(df.tail())

print("\n3. DATASET SHAPE")
print(df.shape)

print("\n4. COLUMN NAMES")
print(df.columns.tolist())

print("\n5. DATASET INFORMATION")
df.info()

# ------------------------------------------------
# STEP 3: STATISTICAL INFORMATION
# ------------------------------------------------

print("\n6. DESCRIPTIVE STATISTICS")
print(df.describe())

print("\n7. DESCRIPTIVE STATISTICS INCLUDING CATEGORICAL")
print(df.describe(include="all"))

# ------------------------------------------------
# STEP 4: DATA QUALITY CHECK
# ------------------------------------------------

print("\n8. MISSING VALUES")
print(df.isnull().sum())

print("\n9. DUPLICATE RECORDS")
print(df.duplicated().sum())

print("\n10. UNIQUE VALUES")
for column in df.columns:
    print(column, ":", df[column].nunique())

# ------------------------------------------------
# STEP 5: DATA TYPES
# ------------------------------------------------

print("\n11. DATA TYPES")
print(df.dtypes)

print("\n========== INITIAL INSPECTION COMPLETED ==========")
import pandas as pd
import numpy as np

# ==========================================
# STEP 3: DATA CLEANING
# ==========================================

df_clean = df.copy()

print("\n========== DATA CLEANING STARTED ==========")

# 1. Check Missing Values
print("\n1. Missing Values Before Cleaning:")
print(df_clean.isnull().sum())

# 2. Remove Duplicate Records
print("\n2. Duplicate Records Before Cleaning:")
print(df_clean.duplicated().sum())

df_clean = df_clean.drop_duplicates()

print("Duplicate Records After Cleaning:")
print(df_clean.duplicated().sum())

# 3. Handle Missing Values
for column in df_clean.columns:
    if df_clean[column].isnull().sum() > 0:
        if pd.api.types.is_numeric_dtype(df_clean[column]):
            # Fill numerical columns with median
            df_clean[column] = df_clean[column].fillna(
                df_clean[column].median()
            )
        else:
            # Fill categorical columns with mode
            df_clean[column] = df_clean[column].fillna(
                df_clean[column].mode()[0]
            )
print("\nMissing Values After Cleaning:")
print(df_clean.isnull().sum())
print("\nDuplicate Records After Cleaning:")
print(df_clean.duplicated().sum())

# 4. Remove Extra Spaces from Categorical Values
categorical_columns = df_clean.select_dtypes(
    include="object"
).columns
for column in categorical_columns:
    df_clean[column] = df_clean[column].str.strip().str.lower()

# 5. Check Data Types
print("\n3. Data Types:")
print(df_clean.dtypes)

# 6. Validate Numerical Columns
print("\n4. Numerical Data Validation:")
print(df_clean.describe())

# 7. Final Missing Value Check
print("\n5. Missing Values After Cleaning:")
print(df_clean.isnull().sum())

# 8. Final Dataset Shape
print("\n6. Final Dataset Shape:")
print(df_clean.shape)

# 9. Save Cleaned Dataset
df_clean.to_csv(
    "datasets/insurance_cleaned.csv",
    index=False
)
print("\nCleaned Dataset Saved Successfully!")
print("\n========== DATA CLEANING COMPLETED ==========")

import matplotlib.pyplot as plt
import seaborn as sns
import os
# Create visualization folder
os.makedirs("visualizations", exist_ok=True)
sns.set_style("whitegrid")
print("\n========== EXPLORATORY DATA ANALYSIS ==========")

# -----------------------------------------------------
# 1. Age Distribution
# -----------------------------------------------------
plt.figure(figsize=(9, 5))
sns.histplot(df_clean["age"], bins=20, kde=True)
plt.title("Age Distribution of Insurance Customers")
plt.xlabel("Age")
plt.ylabel("Number of Customers")
plt.savefig("visualizations/age_distribution.png")
plt.show()

# -----------------------------------------------------
# 2. Insurance Charges Distribution
# -----------------------------------------------------
plt.figure(figsize=(9, 5))
sns.histplot(df_clean["charges"], bins=30, kde=True)
plt.title("Distribution of Insurance Charges")
plt.xlabel("Insurance Charges")
plt.ylabel("Frequency")
plt.savefig("visualizations/charges_distribution.png")
plt.show()

# -----------------------------------------------------
# 3. BMI Distribution
# -----------------------------------------------------
plt.figure(figsize=(9, 5))
sns.histplot(df_clean["bmi"], bins=25, kde=True)
plt.title("BMI Distribution")
plt.xlabel("BMI")
plt.ylabel("Frequency")
plt.savefig("visualizations/bmi_distribution.png")
plt.show()

# -----------------------------------------------------
# 4. Smoking Status vs Insurance Charges
# -----------------------------------------------------
plt.figure(figsize=(8, 5))
sns.boxplot(data=df_clean, x="smoker", y="charges")
plt.title("Smoking Status vs Insurance Charges")
plt.xlabel("Smoker")
plt.ylabel("Insurance Charges")
plt.savefig("visualizations/smoker_vs_charges.png")
plt.show()

# -----------------------------------------------------
# 5. Gender vs Insurance Charges
# -----------------------------------------------------
plt.figure(figsize=(8, 5))
sns.boxplot(data=df_clean, x="sex", y="charges")
plt.title("Gender vs Insurance Charges")
plt.xlabel("Gender")
plt.ylabel("Insurance Charges")
plt.savefig("visualizations/gender_vs_charges.png")
plt.show()

# -----------------------------------------------------
# 6. Region vs Insurance Charges
# -----------------------------------------------------
plt.figure(figsize=(9, 5))
sns.boxplot(data=df_clean, x="region", y="charges")
plt.title("Regional Insurance Charges Analysis")
plt.xlabel("Region")
plt.ylabel("Insurance Charges")
plt.savefig("visualizations/region_vs_charges.png")
plt.show()

# -----------------------------------------------------
# 7. Correlation Heatmap
# -----------------------------------------------------
plt.figure(figsize=(9, 6))
numerical_df = df_clean.select_dtypes(include=np.number)
correlation = numerical_df.corr()
sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)
plt.title("Correlation Between Numerical Features")
plt.savefig("visualizations/correlation_heatmap.png")
plt.show()

# -----------------------------------------------------
# 8. Age vs Insurance Charges
# -----------------------------------------------------
plt.figure(figsize=(9, 5))
sns.scatterplot(
    data=df_clean,
    x="age",
    y="charges",
    hue="smoker"
)
plt.title("Age vs Insurance Charges")
plt.xlabel("Age")
plt.ylabel("Insurance Charges")
plt.savefig("visualizations/age_vs_charges.png")
plt.show()
print("\nEDA COMPLETED SUCCESSFULLY!")
print("\nVisualizations saved inside visualizations folder.")

# =====================================================
# STEP 5: STATISTICAL ANALYSIS & BUSINESS INSIGHTS
# ====================================================

print("\n========== STATISTICAL ANALYSIS ==========")
import os
os.makedirs("reports", exist_ok=True)

# 1. Smoking Status Analysis
smoking_analysis = df_clean.groupby("smoker")["charges"].agg(
    ["count", "mean", "median", "min", "max"]
)
print("\n1. SMOKING STATUS ANALYSIS")
print(smoking_analysis)

# 2. Gender Analysis
gender_analysis = df_clean.groupby("sex")["charges"].mean()
print("\n2. GENDER-WISE AVERAGE CHARGES")
print(gender_analysis)

# 3. Regional Analysis
region_analysis = df_clean.groupby("region")["charges"].agg(
    ["mean", "median", "count"]
)
print("\n3. REGION-WISE ANALYSIS")
print(region_analysis)

# 4. Age Group Analysis
df_clean["age_group"] = pd.cut(
    df_clean["age"],
    bins=[17, 25, 35, 45, 55, 65],
    labels=["18-25", "26-35", "36-45", "46-55", "56-64"]
)
age_analysis = df_clean.groupby(
    "age_group", observed=False
)["charges"].mean()
print("\n4. AGE GROUP ANALYSIS")
print(age_analysis)

# 5. BMI Category Analysis
df_clean["bmi_category"] = pd.cut(
    df_clean["bmi"],
    bins=[0, 18.5, 25, 30, float("inf")],
    labels=["Underweight", "Normal", "Overweight", "Obese"]
)
bmi_analysis = df_clean.groupby(
    "bmi_category", observed=False
)["charges"].mean()
print("\n5. BMI CATEGORY ANALYSIS")
print(bmi_analysis)

# 6. Correlation with Insurance Charges
correlation = df_clean.select_dtypes(
    include=np.number
).corr()
charge_correlation = correlation["charges"].sort_values(
    ascending=False
)
print("\n6. CORRELATION WITH INSURANCE CHARGES")
print(charge_correlation)

# 7. Save Statistical Reports
smoking_analysis.to_csv("reports/smoking_analysis.csv")
region_analysis.to_csv("reports/region_analysis.csv")
age_analysis.to_csv("reports/age_analysis.csv")
bmi_analysis.to_csv("reports/bmi_analysis.csv")
charge_correlation.to_csv("reports/charge_correlation.csv")
print("\nSTATISTICAL ANALYSIS COMPLETED!")
print("\nReports saved successfully inside reports folder.")

# =====================================================
# STEP 6: FEATURE ENGINEERING & DATA PREPARATION
# =====================================================
from sklearn.model_selection import train_test_split
print("\n========== FEATURE ENGINEERING ==========")
# -----------------------------------------------------
# 1. Create Feature Categories
# -----------------------------------------------------
df_model = df_clean.copy()
print("\nOriginal Dataset Shape:", df_model.shape)
# -----------------------------------------------------
# 2. Encode Categorical Variables
# -----------------------------------------------------
categorical_columns = [
    "sex",
    "smoker",
    "region",
    "age_group",
    "bmi_category"
]
df_encoded = pd.get_dummies(
    df_model,
    columns=categorical_columns,
    drop_first=True,
    dtype=int
)
print("\nCategorical Encoding Completed")
# -----------------------------------------------------
# 3. Separate Features and Target
# -----------------------------------------------------
X = df_encoded.drop(columns=["charges"])
y = df_encoded["charges"]
print("\nIndependent Features Shape:", X.shape)
print("\nTarget Variable Shape:", y.shape)
# -----------------------------------------------------
# 4. Split Training and Testing Data
# -----------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)
print("\nTraining Data Shape:", X_train.shape)
print("\nTesting Data Shape:", X_test.shape)
# -----------------------------------------------------
# 5. Save Prepared Dataset
# -----------------------------------------------------
os.makedirs("datasets/processed", exist_ok=True)
X_train.to_csv(
    "datasets/processed/X_train.csv",
    index=False
)
X_test.to_csv(
    "datasets/processed/X_test.csv",
    index=False
)
y_train.to_csv(
    "datasets/processed/y_train.csv",
    index=False
)
y_test.to_csv(
    "datasets/processed/y_test.csv",
    index=False
)
# Save training feature columns

import joblib
import os

os.makedirs("models", exist_ok=True)

joblib.dump(
    list(X_train.columns),
    "models/feature_columns.pkl"
)

print("Feature columns saved successfully!")
print("\n========== FEATURE ENGINEERING COMPLETED ==========")
# =====================================================
# STEP 7: MACHINE LEARNING MODEL TRAINING
# =====================================================

import joblib
import os

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor

print("\n========== MODEL TRAINING STARTED ==========")

os.makedirs("models", exist_ok=True)

# -----------------------------------------------------
# 1. Initialize Models
# -----------------------------------------------------
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(
        max_depth=5,
        random_state=42
    ),
    "Random Forest": RandomForestRegressor(
        n_estimators=200,
        max_depth=8,
        random_state=42
    ),
    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    )
}

# 2. Train Models
trained_models = {}
for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)
    trained_models[name] = model
    print(f"{name} trained successfully!")

# 3. Save Trained Models
for name, model in trained_models.items():
    filename = name.lower().replace(" ", "_")
    joblib.dump(
        model,
        f"models/{filename}.pkl"
    )
    print(f"{name} saved successfully!")

# 4. Save Feature Names
joblib.dump(
    X_train.columns.tolist(),
    "models/feature_names.pkl"
)
print("\n========== ALL MODELS TRAINED ==========")
import pandas as pd
import numpy as np
import joblib

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
results = []

# Evaluate each model
for name, model in models.items():
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predictions)
    results.append({
        "Model": name,
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2 Score": r2
    })

# Create comparison table
results_df = pd.DataFrame(results)
print("\nMODEL EVALUATION RESULTS")
print(results_df)

# Save results
results_df.to_csv(
    "reports/model_evaluation_results.csv",
    index=False
)
print("\nEvaluation completed successfully!")
# STEP 9: MODEL COMPARISON

import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

# Display model comparison
print("\n===== MODEL COMPARISON =====")

print(results_df.sort_values(by="RMSE"))

# Select best model based on lowest RMSE
best_model_row = results_df.loc[
    results_df["RMSE"].idxmin()
]

best_model_name = best_model_row["Model"]

print("\n===== BEST MODEL =====")
print("Best Model:", best_model_name)
print("RMSE:", best_model_row["RMSE"])
print("R2 Score:", best_model_row["R2 Score"])

# Save best model
best_model = models[best_model_name]

os.makedirs("models", exist_ok=True)

joblib.dump(best_model, "models/best_model.pkl")

print("\nBest model saved successfully!")

# Visual comparison
plt.figure(figsize=(10, 6))

sns.barplot(
    data=results_df,
    x="Model",
    y="RMSE"
)
plt.title("Model Comparison - RMSE")
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig("model_comparison.png")
plt.show()
print("\nStep 9 completed!")
# STEP 10: ACTUAL VS PREDICTED

import matplotlib.pyplot as plt
import seaborn as sns
import os

# Generate predictions
best_model = models["Gradient Boosting"]

y_pred = best_model.predict(X_test)

# Create visualization
plt.figure(figsize=(10, 6))

sns.scatterplot(
    x=y_test,
    y=y_pred,
    alpha=0.6
)

# Perfect prediction reference line
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    linestyle="--"
)

plt.xlabel("Actual Insurance Charges")
plt.ylabel("Predicted Insurance Charges")

plt.title("Actual vs Predicted Insurance Charges")

plt.tight_layout()

os.makedirs("visualizations", exist_ok=True)

plt.savefig(
    "visualizations/actual_vs_predicted.png"
)

plt.show()
print("Actual vs Predicted visualization completed!")
# STEP 11: RESIDUAL ANALYSIS

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Calculate residuals
residuals = y_test - y_pred

# Create directory
os.makedirs("visualizations", exist_ok=True)

# 1. Residual Distribution
plt.figure(figsize=(10, 6))

sns.histplot(
    residuals,
    kde=True,
    bins=30
)

plt.axvline(0, linestyle="--")

plt.title("Residual Distribution")
plt.xlabel("Prediction Error")
plt.ylabel("Frequency")

plt.tight_layout()

plt.savefig(
    "visualizations/residual_distribution.png"
)

plt.show()


# 2. Residuals vs Predicted Values
plt.figure(figsize=(10, 6))

sns.scatterplot(
    x=y_pred,
    y=residuals,
    alpha=0.6
)

plt.axhline(0, linestyle="--")

plt.title("Residuals vs Predicted Values")
plt.xlabel("Predicted Insurance Charges")
plt.ylabel("Residuals")

plt.tight_layout()

plt.savefig(
    "visualizations/residuals_vs_predicted.png"
)

plt.show()


# Print residual statistics
print("\n===== RESIDUAL ANALYSIS =====")

print("Mean Residual:", np.mean(residuals))
print("Maximum Residual:", np.max(residuals))
print("Minimum Residual:", np.min(residuals))
print("Residual Standard Deviation:", np.std(residuals))

print("\nResidual Analysis Completed!")

# STEP 12: FEATURE IMPORTANCE ANALYSIS

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Get best model
best_model = models["Gradient Boosting"]

# Extract feature importance
importance = best_model.feature_importances_

# Get feature names
feature_names = X_test.columns

# Create dataframe
feature_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importance
})

# Sort features
feature_df = feature_df.sort_values(
    by="Importance",
    ascending=False
)

print("\n===== FEATURE IMPORTANCE =====")
print(feature_df)

# Save results
os.makedirs("reports", exist_ok=True)

feature_df.to_csv(
    "reports/feature_importance.csv",
    index=False
)

# Visualization
plt.figure(figsize=(10, 6))

sns.barplot(
    data=feature_df,
    x="Importance",
    y="Feature"
)

plt.title("Feature Importance - Gradient Boosting")
plt.xlabel("Importance Score")
plt.ylabel("Features")

plt.tight_layout()

plt.savefig(
    "visualizations/feature_importance.png"
)

plt.show()

print("\nFeature Importance Analysis Completed!")
# =====================================================
# STEP 13: PREDICTION TESTING
# =====================================================

import pandas as pd
import numpy as np
import joblib
import os

# Load best trained model
best_model = joblib.load("models/best_model.pkl")

# Generate predictions
predictions = best_model.predict(X_test)

# Create prediction comparison dataframe
prediction_results = pd.DataFrame({

    "Actual Charges": np.array(y_test).flatten(),

    "Predicted Charges": predictions,

})

# Calculate prediction error
prediction_results["Prediction Error"] = (
    prediction_results["Actual Charges"]
    - prediction_results["Predicted Charges"]
)

prediction_results["Absolute Error"] = (
    prediction_results["Prediction Error"].abs()
)

# Display first 15 predictions
print("\n===== PREDICTION TESTING =====")

print(prediction_results.head(15))

# Calculate average prediction error
print("\n===== PREDICTION SUMMARY =====")
print(
    "Average Absolute Error:",
    prediction_results["Absolute Error"].mean()
)
print(
    "Maximum Prediction Error:",
    prediction_results["Absolute Error"].max()
)
print(
    "Minimum Prediction Error:",
    prediction_results["Absolute Error"].min()
)
# Save prediction results
os.makedirs("reports", exist_ok=True)
prediction_results.to_csv(
    "reports/prediction_results.csv",
    index=False
)
print("\nPrediction results saved successfully!")
print("\nSTEP 13 COMPLETED!")
# =====================================================
# STEP 14: CUSTOM INSURANCE COST PREDICTION
# =====================================================

import pandas as pd
import joblib

# Load best model
best_model = joblib.load("models/best_model.pkl")

print("\n===================================")
print(" HEALTH INSURANCE COST PREDICTION")
print("===================================")

# Take user input
age = int(input("Enter your age: "))

sex = input("Enter gender (male/female): ").lower()

bmi = float(input("Enter BMI: "))

children = int(input("Enter number of children: "))

smoker = input("Are you a smoker? (yes/no): ").lower()

region = input(
    "Enter region (southwest/southeast/northwest/northeast): "
).lower()

# Create input dataframe
user_data = pd.DataFrame({
    "age": [age],
    "sex": [sex],
    "bmi": [bmi],
    "children": [children],
    "smoker": [smoker],
    "region": [region]
})

# Apply same one-hot encoding used during training
user_encoded = pd.get_dummies(user_data)

# Match training feature columns
user_encoded = user_encoded.reindex(
    columns=X_train.columns,
    fill_value=0
)

# Predict insurance charges
predicted_cost = best_model.predict(user_encoded)

print("\n===================================")
print("       PREDICTION RESULT")
print("===================================")

print(
    f"Estimated Insurance Cost: ₹{predicted_cost[0]:,.2f}"
)

print("\nPrediction completed successfully!")