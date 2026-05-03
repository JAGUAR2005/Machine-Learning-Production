import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, TargetEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    r2_score, mean_absolute_error, mean_squared_error, 
    accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
)
import joblib
import os
import json
from datetime import datetime

# Create plots directory
os.makedirs('plots/logistic', exist_ok=True)

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
            if 'Lakh' in x:
                return float(x.replace('Lakh', '').strip()) * 100000 * 0.012
            elif 'Crore' in x:
                return float(x.replace('Crore', '').strip()) * 10000000 * 0.012
            else:
                return float(x) * 0.012
        except:
            return np.nan
            
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
print("  RESALE INTELLIGENCE — Logistic/Linear Regression v1.0")
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
df_merged['km_per_year'] = df_merged['mileage'] / (df_merged['car_age'] + 1)
df_merged['fuel_type'] = df_merged['fuel_type'].apply(lambda x: 'diesel' if 'diesel' in str(x) else ('petrol' if 'petrol' in str(x) or 'gasoline' in str(x) else ('electric' if 'electric' in str(x) else ('hybrid' if 'hybrid' in str(x) else 'other'))))
df_merged['transmission'] = df_merged['transmission'].apply(lambda x: 'automatic' if 'auto' in str(x) else ('manual' if 'manual' in str(x) else 'other'))
df_merged['log_mileage'] = np.log1p(df_merged['mileage'])
df_merged['mileage_age_ratio'] = df_merged['mileage'] / (df_merged['car_age'] + 0.5)
df_merged['age_squared'] = df_merged['car_age'] ** 2
df_merged['price_log'] = np.log1p(df_merged['price'])
luxury_brands = {'bmw', 'mercedes-benz', 'merc', 'audi', 'jaguar', 'land', 'porsche', 'volvo', 'lexus', 'maserati', 'bentley', 'ferrari', 'lamborghini', 'aston-martin', 'mini'}
df_merged['is_luxury'] = df_merged['brand'].isin(luxury_brands).astype(int)
df_merged['mileage_bucket'] = pd.cut(df_merged['mileage'], bins=[0, 15000, 50000, 100000, 150000, 300000], labels=['like_new', 'low', 'moderate', 'high', 'very_high']).astype(str)
df_merged['price_segment'] = pd.qcut(df_merged['price'], q=4, labels=['Budget', 'Mid-Range', 'Premium', 'Luxury'])
segment_mapping = {'Budget': 0, 'Mid-Range': 1, 'Premium': 2, 'Luxury': 3}
df_merged['price_segment_int'] = df_merged['price_segment'].map(segment_mapping)

print(f"  Total records: {len(df_merged)}")

# Preparation
X = df_merged[['brand', 'model', 'car_age', 'mileage', 'km_per_year', 'log_mileage', 'mileage_age_ratio', 'age_squared', 'is_luxury', 'fuel_type', 'transmission', 'market', 'mileage_bucket']]
y_reg = df_merged['price_log']
y_clf = df_merged['price_segment_int']

categorical_features = ['fuel_type', 'transmission', 'market', 'mileage_bucket']
high_cardinality_features = ['brand', 'model']
numeric_features = ['car_age', 'mileage', 'km_per_year', 'log_mileage', 'mileage_age_ratio', 'age_squared', 'is_luxury']

preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(), numeric_features),
    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features),
    ('target', TargetEncoder(smooth='auto'), high_cardinality_features)
])

X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test = train_test_split(X, y_reg, y_clf, test_size=0.2, random_state=42, stratify=y_clf)

print("\n--- PHASE 2: Linear Regression (Price Prediction) ---")
reg_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', LinearRegression())
])
reg_pipeline.fit(X_train, y_reg_train)

y_reg_pred_log = reg_pipeline.predict(X_test)
y_reg_pred = np.expm1(y_reg_pred_log)
y_reg_test_orig = np.expm1(y_reg_test)

r2 = r2_score(y_reg_test_orig, y_reg_pred)
mae = mean_absolute_error(y_reg_test_orig, y_reg_pred)
print(f"  Linear Regression R2: {r2:.4f}, MAE: ${mae:.2f}")

print("\n--- PHASE 3: Logistic Regression (Classification) ---")
# Tuning Logistic Regression for better accuracy
clf_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', LogisticRegression(max_iter=1000, multi_class='multinomial'))
])

param_grid_clf = {
    'model__C': [0.1, 1.0, 10.0],
    'model__solver': ['lbfgs', 'saga']
}

print("  Searching for best Logistic Regression parameters...")
clf_search = RandomizedSearchCV(clf_pipeline, param_grid_clf, n_iter=5, cv=3, scoring='accuracy', random_state=42)
clf_search.fit(X_train, y_clf_train)
best_clf_pipeline = clf_search.best_estimator_

y_clf_pred = best_clf_pipeline.predict(X_test)
accuracy = accuracy_score(y_clf_test, y_clf_pred)
print(f"  Logistic Regression Accuracy: {accuracy:.4f}")
print(f"  Best Params: {clf_search.best_params_}")

print("\n--- PHASE 4: Visualizations ---")
target_names = ['Budget', 'Mid-Range', 'Premium', 'Luxury']

# 1. Predicted vs Actual
plt.figure(figsize=(10, 6))
plt.scatter(y_reg_test_orig, y_reg_pred, alpha=0.3, color='blue', s=5)
plt.plot([y_reg_test_orig.min(), y_reg_test_orig.max()], [y_reg_test_orig.min(), y_reg_test_orig.max()], 'r--', lw=2)
plt.title(f'Linear Regression: Predicted vs Actual (R²={r2:.4f})')
plt.savefig('plots/logistic/predicted_vs_actual.png', dpi=150)
plt.close()

# 2. Confusion Matrix
conf_matrix = confusion_matrix(y_clf_test, y_clf_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
plt.title('Logistic Regression Confusion Matrix')
plt.savefig('plots/logistic/confusion_matrix.png', dpi=150)
plt.close()

# 3. Coefficients (Feature Importance Proxy)
plt.figure(figsize=(10, 6))
# For regression model
coeffs = reg_pipeline.named_steps['model'].coef_
feature_names = reg_pipeline.named_steps['preprocessor'].get_feature_names_out()
top_idx = np.argsort(np.abs(coeffs))[-10:]
plt.barh(range(10), coeffs[top_idx], color='green')
plt.yticks(range(10), [str(feature_names[i])[:25] for i in top_idx])
plt.title('Top 10 Linear Regression Coefficients')
plt.savefig('plots/logistic/feature_importance.png', dpi=150)
plt.close()

print("  Plots saved to plots/logistic/")

# Save metrics for comparison
metrics = {
    "algorithm": "Logistic/Linear Regression",
    "r2": r2,
    "mae": mae,
    "accuracy": accuracy,
    "timestamp": datetime.now().isoformat()
}
with open('metrics_logistic.json', 'w') as f:
    json.dump(metrics, f, indent=2)

print("\n--- PHASE 5: Saving Models ---")
joblib.dump(reg_pipeline, 'logistic_reg_pipeline.pkl')
joblib.dump(best_clf_pipeline, 'logistic_clf_pipeline.pkl')
print("  Models saved as logistic_reg_pipeline.pkl and logistic_clf_pipeline.pkl")
