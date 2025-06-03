# model/train_model.py
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OrdinalEncoder
import numpy as np
import os

# mapping for OrdinalEncoder
position_order = [
    "Staff", "Supervisor",
    "Department Head", "Deputy Department Head", "Division Head", "Vice Division Head",
    "Secretary", "Treasurer", "Vice Chair", "Chair"
]
chat_activity_order = [
    "Inactive", "Rarely Active", "Moderate", "Active", "Very Active"
]

# Load the dataset
df = pd.read_csv("data/activity_dataset.csv")

# Convert fractional fields to percentage
def convert_fraction(col_series):
    return col_series.apply(lambda x: round(eval(str(x)), 2) if isinstance(x, str) and '/' in str(x) else x)

df["event_attendance"] = convert_fraction(df["event_attendance"])
df["meeting_attendance"] = convert_fraction(df["meeting_attendance"])
df["duty_participation"] = convert_fraction(df["duty_participation"])

# Handle potential missing values in converted columns
df["event_attendance"] = df["event_attendance"].fillna(0)
df["meeting_attendance"] = df["meeting_attendance"].fillna(0)
df["duty_participation"] = df["duty_participation"].fillna(0)

# Encode categorical features using OrdinalEncoder
oe_position = OrdinalEncoder(categories=[position_order], handle_unknown='use_encoded_value', unknown_value=-1)
df["position_encoded"] = oe_position.fit_transform(df[["position"]])

oe_chat = OrdinalEncoder(categories=[chat_activity_order], handle_unknown='use_encoded_value', unknown_value=-1)
df["chat_activity_encoded"] = oe_chat.fit_transform(df[["chat_activity"]])


# Normalisasi 'submitted_works' dan 'competition_participation'
max_submitted_works = df["submitted_works"].max()
max_competition_participation = df["competition_participation"].max()

df["submitted_works_norm"] = df["submitted_works"] / (max_submitted_works if max_submitted_works > 0 else 1)
df["competition_participation_norm"] = df["competition_participation"] / (max_competition_participation if max_competition_participation > 0 else 1)

df["submitted_works_norm"] = df["submitted_works_norm"].fillna(0)
df["competition_participation_norm"] = df["competition_participation_norm"].fillna(0)

df['chat_activity_scaled_0_1'] = df['chat_activity_encoded'] / (len(chat_activity_order) - 1) # Skala 0-4 ke 0-1
df['position_scaled_0_1'] = df['position_encoded'] / (len(position_order) - 1) # Skala 0-3 ke 0-1

df["activity_score"] = (
    0.20 * df["event_attendance"] +
    0.15 * df["meeting_attendance"] +
    0.10 * df["duty_participation"] +
    0.15 * df["submitted_works_norm"] +
    0.20 * df["competition_participation_norm"] +
    0.10 * df["chat_activity_scaled_0_1"] + 
    0.10 * df["position_scaled_0_1"]        
)

# Ambil min/max skor mentah dari data yang dihasilkan
min_score_raw = df["activity_score"].min()
max_score_raw = df["activity_score"].max()

print(f"DEBUG TRAIN: Raw min activity score (from data) NEW FORMULA: {min_score_raw:.4f}")
print(f"DEBUG TRAIN: Raw max activity score (from data) NEW FORMULA: {max_score_raw:.4f}")

# Scaling activity_score ke rentang 0-100 untuk training
# Gunakan rentang sebenarnya dari data, tetapi dengan sedikit "padding"
# Atau, jika range_diff sangat kecil, kita bisa atur min_score_raw menjadi 0.0 dan max_score_raw menjadi 1.0 (jika memang itu rentang idealnya)
effective_min_score = min_score_raw
effective_max_score = max_score_raw

# Jika rentang terlalu sempit, berikan rentang minimal yang wajar (misal 0.0-1.0)
if (effective_max_score - effective_min_score) < 0.01: # Threshold yang lebih kecil
    effective_min_score = 0.0
    effective_max_score = 1.0 # Ini adalah target skor maksimal kita setelah penskalaan internal 0-1
    print("Warning: Activity score range too narrow. Using 0.0-1.0 for scaling target in training.")

# Scaling ke 0-100
df["activity_score_scaled"] = ((df["activity_score"] - effective_min_score) / (effective_max_score - effective_min_score)) * 100

# Pastikan skor tetap dalam 0-100 setelah scaling
df["activity_score_scaled"] = np.clip(df["activity_score_scaled"], 0, 100)

print(f"DEBUG TRAIN: Scaled min activity score (for training): {df['activity_score_scaled'].min():.2f}")
print(f"DEBUG TRAIN: Scaled max activity score (for training): {df['activity_score_scaled'].max():.2f}")


# Define features and target (target sekarang adalah activity_score_scaled)
features = [
    "position_encoded", "event_attendance", "meeting_attendance",
    "duty_participation", "submitted_works_norm", "competition_participation_norm",
    "chat_activity_scaled_0_1", # Gunakan versi yang sudah diskalakan 0-1 sebagai fitur input
    "position_scaled_0_1"       # Gunakan versi yang sudah diskalakan 0-1 sebagai fitur input
]
X = df[features]
y = df["activity_score_scaled"] # Menggunakan target yang sudah diskalakan

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model (RandomForestRegressor)
model = RandomForestRegressor(n_estimators=100, random_state=42, oob_score=True, 
                              max_depth=10, # Jaga hyperparameter ini
                              min_samples_leaf=5) # Jaga hyperparameter ini
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred):.2f}")
print(f"R2 Score: {r2_score(y_test, y_pred):.2f}")

# Get Feature Importance
feature_importances = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
print("\nFeature Importances:")
print(feature_importances)

# Save the model, encoders, and feature importances
joblib.dump(model, "model/model_regressor.pkl")
joblib.dump(oe_position, "model/oe_position.pkl") 
joblib.dump(oe_chat, "model/oe_chat.pkl") 
joblib.dump(feature_importances, "model/feature_importances.pkl")
joblib.dump(max_submitted_works, "model/max_submitted_works.pkl")
joblib.dump(max_competition_participation, "model/max_competition_participation.pkl")
joblib.dump(min_score_raw, "model/min_activity_score.pkl") # TETAP simpan min_score_raw
joblib.dump(max_score_raw, "model/max_activity_score.pkl") # TETAP simpan max_score_raw (dari data mentah)

print("Model, encoders, dan feature importances telah disimpan.")