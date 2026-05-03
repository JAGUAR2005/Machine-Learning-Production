import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, TargetEncoder
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier, plot_tree
from sklearn.metrics import r2_score, mean_absolute_error, accuracy_score, confusion_matrix
import joblib
import os
import json
from datetime import datetime

# Create plots directory
os.makedirs('plots/decision_tree', exist_ok=True)

def clean_india(df):
    def extract_brand(x):
        parts = str(x).split(' ')
        return parts[1].lower() if len(parts) > 1 else 'unknown'
    def extract_model(x):
        parts = str(x).split(' ')
        return parts[2].lower() if len(parts) > 2 else 'other'
    df['brand'] = df['full_name'].apply(extract_brand)
    df['model'] = df['full_name'].apply(extract_model)
    def parse_price(x):
        try:
            x = str(x).replace('₹', '').replace(',', '').strip()
            if 'Lakh' in x: return float(x.replace('Lakh', '').strip()) * 100000 * 0.012
            elif 'Crore' in x: return float(x.replace('Crore', '').strip()) * 10000000 * 0.012
            else: return float(x) * 0.012
        except: return np.nan
    df['price'] = df['resale_price'].apply(parse_price)
    df['mileage'] = df['kms_driven'].astype(str).str.replace(',', '').str.replace(' Kms', '').str.replace(' km', '').str.extract(r'(\d+)').astype(float)
    df['year'] = pd.to_numeric(df['registered_year'], errors='coerce')
    df['fuel_type'] = df['fuel_type'].str.lower()
    df['transmission'] = df['transmission_type'].str.lower()
    df['market'] = 'india'
    return df[['brand', 'model', 'year', 'mileage', 'fuel_type', 'transmission', 'market', 'price']].dropna()

def clean_europe(df):
    df['brand'] = df['brand'].str.lower()
    df['model'] = df['model'].str.lower()
    df['price'] = pd.to_numeric(df['price_in_euro'], errors='coerce') * 1.08
    df['mileage'] = pd.to_numeric(df['mileage_in_km'], errors='coerce')
    df['year'] = pd.to_numeric(df['year'], errors='coerce')
    df['fuel_type'] = df['fuel_type'].str.lower()
    df['transmission'] = df['transmission_type'].str.lower()
    df['market'] = 'europe'
    return df[['brand', 'model', 'year', 'mileage', 'fuel_type', 'transmission', 'market', 'price']].dropna()

def clean_uk(df_list):
    dfs = []
    for f in df_list:
        df = pd.read_csv(f)
        brand = f.split('/')[-1].replace('.csv', '').replace('unclean ', '').lower()
        df['brand'] = brand
        df['model'] = df['model'].str.lower()
        df['price'] = pd.to_numeric(df['price'], errors='coerce') * 1.26
        df['mileage'] = pd.to_numeric(df['mileage'], errors='coerce')
        df['year'] = pd.to_numeric(df['year'], errors='coerce')
        df['fuel_type'] = df.get('fuelType', df.get('fuel type')).str.lower()
        df['transmission'] = df['transmission'].str.lower()
        df['market'] = 'asia_uk'
        dfs.append(df[['brand', 'model', 'year', 'mileage', 'fuel_type', 'transmission', 'market', 'price']].dropna())
    return pd.concat(dfs)

print("=" * 60)
print("  RESALE INTELLIGENCE — Decision Tree v1.0")
print("=" * 60)

print("\n--- PHASE 1: Data Collection & Cleaning ---")
df_india = pd.read_csv('car_resale_prices.csv')
india_clean = clean_india(df_india)
df_europe = pd.read_csv('data.csv')
europe_clean = clean_europe(df_europe)
uk_files = glob.glob('archive/*.csv')
uk_clean = clean_uk(uk_files)

df_merged = pd.concat([india_clean, europe_clean, uk_clean], ignore_index=True)
df_merged = df_merged[(df_merged['price'] > 500) & (df_merged['price'] < 150000)]
df_merged = df_merged[(df_merged['year'] >= 2000) & (df_merged['year'] <= 2025)]
df_merged = df_merged[(df_merged['mileage'] < 300000) & (df_merged['mileage'] >= 0)]

# Feature Engineering
current_year = datetime.now().year
df_merged['car_age'] = current_year - df_merged['year']
df_merged['price_log'] = np.log1p(df_merged['price'])
df_merged['price_segment'] = pd.qcut(df_merged['price'], q=4, labels=['Budget', 'Mid-Range', 'Premium', 'Luxury'])
segment_mapping = {'Budget': 0, 'Mid-Range': 1, 'Premium': 2, 'Luxury': 3}
df_merged['price_segment_int'] = df_merged['price_segment'].map(segment_mapping)

# Preparation
X = df_merged[['brand', 'model', 'car_age', 'mileage', 'fuel_type', 'transmission', 'market']]
y_reg = df_merged['price_log']
y_clf = df_merged['price_segment_int']

categorical_features = ['fuel_type', 'transmission', 'market']
high_cardinality_features = ['brand', 'model']
numeric_features = ['car_age', 'mileage']

preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(), numeric_features),
    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features),
    ('target', TargetEncoder(smooth='auto'), high_cardinality_features)
])

X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test = train_test_split(X, y_reg, y_clf, test_size=0.2, random_state=42)

print("\n--- PHASE 2: Decision Tree Regressor ---")
reg_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', DecisionTreeRegressor(max_depth=15, min_samples_leaf=10, random_state=42))
])
reg_pipeline.fit(X_train, y_reg_train)

y_reg_pred_log = reg_pipeline.predict(X_test)
y_reg_pred = np.expm1(y_reg_pred_log)
y_reg_test_orig = np.expm1(y_reg_test)

r2 = r2_score(y_reg_test_orig, y_reg_pred)
mae = mean_absolute_error(y_reg_test_orig, y_reg_pred)
print(f"  Decision Tree Regression R2: {r2:.4f}, MAE: ${mae:.2f}")

print("\n--- PHASE 3: Decision Tree Classifier ---")
clf_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', DecisionTreeClassifier(max_depth=15, min_samples_leaf=10, random_state=42))
])
clf_pipeline.fit(X_train, y_clf_train)

y_clf_pred = clf_pipeline.predict(X_test)
accuracy = accuracy_score(y_clf_test, y_clf_pred)
print(f"  Decision Tree Classification Accuracy: {accuracy:.4f}")

print("\n--- PHASE 4: Decision Tree Visualization ---")
# To make a readable tree, we'll train a small one just for the "Actual Output Tree Map"
viz_tree = DecisionTreeClassifier(max_depth=3, random_state=42)
# We need the numeric names for features
X_train_transformed = preprocessor.fit_transform(X_train, y_clf_train)
# Get feature names from preprocessor
feature_names = preprocessor.get_feature_names_out()
viz_tree.fit(X_train_transformed, y_clf_train)

plt.figure(figsize=(20, 10))
plot_tree(viz_tree, 
          feature_names=feature_names, 
          class_names=['Budget', 'Mid-Range', 'Premium', 'Luxury'],
          filled=True, 
          rounded=True, 
          fontsize=10)
plt.title("Actual Decision Logic: Price Segmentation Tree (First 3 Levels)")
plt.savefig('plots/decision_tree/actual_tree_map.png', dpi=300, bbox_inches='tight')
plt.close()

# Also save a Feature Importance for Decision Tree
plt.figure(figsize=(10, 6))
importances = reg_pipeline.named_steps['model'].feature_importances_
top_idx = np.argsort(importances)[-10:]
plt.barh(range(10), importances[top_idx], color='orange')
plt.yticks(range(10), [str(feature_names[i])[:25] for i in top_idx])
plt.title('Top 10 Decision Tree Feature Importances')
plt.savefig('plots/decision_tree/feature_importance.png', dpi=150)
plt.close()

print("  Decision Tree Map saved to plots/decision_tree/actual_tree_map.png")

# Save metrics
metrics = {
    "algorithm": "Decision Tree",
    "r2": r2,
    "mae": mae,
    "accuracy": accuracy,
    "timestamp": datetime.now().isoformat()
}
with open('metrics_decision_tree.json', 'w') as f:
    json.dump(metrics, f, indent=2)

print("\n--- PHASE 5: Saving Models ---")
joblib.dump(reg_pipeline, 'decision_tree_reg_pipeline.pkl')
joblib.dump(clf_pipeline, 'decision_tree_clf_pipeline.pkl')
print("  Models saved as decision_tree_reg_pipeline.pkl and decision_tree_clf_pipeline.pkl")
