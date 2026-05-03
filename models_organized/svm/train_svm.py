import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, TargetEncoder
from sklearn.svm import LinearSVR, LinearSVC
from sklearn.metrics import r2_score, mean_absolute_error, accuracy_score, confusion_matrix
import joblib
import os
import json
from datetime import datetime
from mpl_toolkits.mplot3d import Axes3D

# Create plots directory
os.makedirs('plots/svm', exist_ok=True)

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
print("  RESALE INTELLIGENCE — Support Vector Machine v1.0")
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
df_merged['price_segment'] = pd.qcut(df_merged['price'], q=4, labels=[0, 1, 2, 3])

# Preparation
X = df_merged[['brand', 'model', 'car_age', 'mileage', 'fuel_type', 'transmission', 'market']]
y_reg = df_merged['price_log']
y_clf = df_merged['price_segment']

categorical_features = ['fuel_type', 'transmission', 'market']
high_cardinality_features = ['brand', 'model']
numeric_features = ['car_age', 'mileage']

preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(), numeric_features),
    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features),
    ('target', TargetEncoder(smooth='auto'), high_cardinality_features)
])

X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test = train_test_split(X, y_reg, y_clf, test_size=0.2, random_state=42)

print("\n--- PHASE 2: Linear SVR (Regression) ---")
# Use a subset for faster SVM training if needed, but LinearSVR is generally fast
reg_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', LinearSVR(dual=True, max_iter=2000, random_state=42))
])
reg_pipeline.fit(X_train, y_reg_train)

y_reg_pred_log = reg_pipeline.predict(X_test)
y_reg_pred = np.expm1(y_reg_pred_log)
y_reg_test_orig = np.expm1(y_reg_test)

r2 = r2_score(y_reg_test_orig, y_reg_pred)
mae = mean_absolute_error(y_reg_test_orig, y_reg_pred)
print(f"  SVM Regression R2: {r2:.4f}, MAE: ${mae:.2f}")

print("\n--- PHASE 3: Linear SVC (Classification) ---")
clf_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', LinearSVC(dual=True, max_iter=2000, random_state=42))
])
clf_pipeline.fit(X_train, y_clf_train)

y_clf_pred = clf_pipeline.predict(X_test)
accuracy = accuracy_score(y_clf_test, y_clf_pred)
print(f"  SVM Classification Accuracy: {accuracy:.4f}")

print("\n--- PHASE 4: High-Quality Plane Visualization ---")
# To visualize a "plane", we need 2 numeric features. Let's use Car Age and Mileage.
# We'll take a sample for the visualization to keep the plot clean
sample_df = df_merged.sample(1000, random_state=42)
X_sample = sample_df[['car_age', 'mileage']]
y_sample = sample_df['price_log']

# Simple SVM for visualization purpose (2D features -> 3D plane)
viz_scaler = StandardScaler()
X_viz_scaled = viz_scaler.fit_transform(X_sample)
viz_model = LinearSVR(dual=True, random_state=42)
viz_model.fit(X_viz_scaled, y_sample)

# Create grid for the plane
x_range = np.linspace(X_viz_scaled[:, 0].min(), X_viz_scaled[:, 0].max(), 20)
y_range = np.linspace(X_viz_scaled[:, 1].min(), X_viz_scaled[:, 1].max(), 20)
xx, yy = np.meshgrid(x_range, y_range)
zz = viz_model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

# Transform back to original scale for plotting
xx_orig = viz_scaler.inverse_transform(np.c_[xx.ravel(), np.zeros_like(xx.ravel())])[:, 0].reshape(xx.shape)
yy_orig = viz_scaler.inverse_transform(np.c_[np.zeros_like(yy.ravel()), yy.ravel()])[:, 1].reshape(yy.shape)

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Scatter plot of actual data
ax.scatter(X_sample['car_age'], X_sample['mileage'], y_sample, color='blue', alpha=0.5, label='Actual Data (Log Price)')

# Plot the SVM Hyperplane
surface = ax.plot_surface(xx_orig, yy_orig, zz, alpha=0.4, cmap='viridis', label='SVM Regression Plane')

ax.set_xlabel('Car Age (Years)')
ax.set_ylabel('Mileage (km)')
ax.set_zlabel('Log(Price)')
ax.set_title('SVM Hyperplane: Depreciation Surface')
plt.savefig('plots/svm/svm_hyperplane_3d.png', dpi=150)
plt.close()

# 2D Decision Boundary (Classification)
plt.figure(figsize=(10, 6))
# Using only 2 features for 2D boundary visualization
viz_model_clf = LinearSVC(dual=True, random_state=42)
y_sample_clf = sample_df['price_segment'].astype(int)
viz_model_clf.fit(X_viz_scaled, y_sample_clf)

h = .02
x_min, x_max = X_viz_scaled[:, 0].min() - 1, X_viz_scaled[:, 0].max() + 1
y_min, y_max = X_viz_scaled[:, 1].min() - 1, X_viz_scaled[:, 1].max() + 1
xx_clf, yy_clf = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = viz_model_clf.predict(np.c_[xx_clf.ravel(), yy_clf.ravel()])
Z = Z.reshape(xx_clf.shape)

plt.contourf(xx_clf, yy_clf, Z, cmap=plt.cm.coolwarm, alpha=0.8)
plt.scatter(X_viz_scaled[:, 0], X_viz_scaled[:, 1], c=y_sample_clf, cmap=plt.cm.coolwarm, edgecolors='k')
plt.xlabel('Car Age (Standardized)')
plt.ylabel('Mileage (Standardized)')
plt.title('SVM Decision Boundaries (Price Segments)')
plt.savefig('plots/svm/svm_decision_boundary.png', dpi=150)
plt.close()

print("  High-quality plane graphs saved to plots/svm/")

# Save metrics
metrics = {
    "algorithm": "Support Vector Machine (Linear)",
    "r2": r2,
    "mae": mae,
    "accuracy": accuracy,
    "timestamp": datetime.now().isoformat()
}
with open('metrics_svm.json', 'w') as f:
    json.dump(metrics, f, indent=2)

print("\n--- PHASE 5: Saving Models ---")
joblib.dump(reg_pipeline, 'svm_reg_pipeline.pkl')
joblib.dump(clf_pipeline, 'svm_clf_pipeline.pkl')
print("  Models saved as svm_reg_pipeline.pkl and svm_clf_pipeline.pkl")
