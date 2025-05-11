import numpy as np
import pandas as pd
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
from tabulate import tabulate
import random

# Reproducibility
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
random.seed(SEED)

# Load data
url = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/2023-24/gws/merged_gw.csv"
df = pd.read_csv(url)

# Group by player to aggregate full-season stats
grouped = df.groupby(['name', 'position', 'team', 'value'], as_index=False).agg({
    "goals_scored": "sum",
    "assists": "sum",
    "clean_sheets": "sum",
    "goals_conceded": "sum",
    "yellow_cards": "sum",
    "minutes": "sum",
    "expected_goals": "sum",
    "expected_assists": "sum",
    "expected_goal_involvements": "sum",
    "xP": "sum",
    "total_points": "sum",
    "starts": "sum",
    "selected": "mean"
})

# Add additional metrics
extra_features = df.groupby('name', as_index=False).agg({
    "ict_index": "sum",
    "bps": "sum",
    "bonus": "sum",
    "creativity": "sum",
    "influence": "sum",
    "threat": "sum",
})

# Merge all together
grouped = pd.merge(grouped, extra_features, on='name', how='left')

# === Features and Target ===
features = [
    "goals_scored", "assists", "clean_sheets", "goals_conceded", "yellow_cards", "minutes",
    "expected_goals", "expected_assists", "expected_goal_involvements", "xP", "starts", 
    "selected", "ict_index", "bps", "bonus", "creativity", "influence", "threat"
]
target = "total_points"

# Log transform skewed features
log_features = ["minutes", "selected", "bps", "influence"]
for feat in log_features:
    grouped[feat] = np.log1p(grouped[feat])

# Prepare data
X = grouped[features].fillna(0)
y = grouped[target]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
joblib.dump(scaler, "scaler.pkl")

# Train/validation split
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=SEED)

# Build the model
input_dim = X_train.shape[1]
model = Sequential([
    Input(shape=(input_dim,)),
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(1)  # Regression output
])
model.compile(optimizer='adam', loss='huber', metrics=['mae'])

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1)

# Train the model
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=32,
    verbose=2,
    callbacks=[early_stopping, lr_scheduler]
)

# Evaluate performance
val_loss, val_mae = model.evaluate(X_val, y_val, verbose=0)
print(f"\nValidation MAE: {val_mae:.2f}")

# Make predictions
y_pred = model.predict(X_scaled).flatten()
y_pred = np.clip(y_pred, 0, None)

# Add predictions
grouped['Predicted Points'] = y_pred
grouped['value'] = (grouped['value'] / 10.0).round(1)

# Sort and save by position
df_sorted = grouped[['name', 'Predicted Points', 'position', 'team', 'value', 'total_points']].sort_values(by='Predicted Points', ascending=False)
positions = ['GK', 'DEF', 'MID', 'FWD']
recommended_players = {}

# Create output directory if not exists
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

for position in positions:
    position_players = df_sorted[df_sorted['position'] == position].drop_duplicates(subset='name')
    recommended_players[position] = position_players[['name', 'position', 'team', 'Predicted Points', 'value']]
    
    # Save each position's CSV to the 'output' folder, overwriting old files
    position_players.to_csv(os.path.join(output_dir, f"{position}_players.csv"), index=False)

    print(f"\nRecommended {position}s:")
    print(tabulate(position_players[['name', 'position', 'team', 'Predicted Points', 'value']], headers='keys', tablefmt='grid', showindex=False))
    print("\n" + "-"*50)

# Final team selection
top_players = {
    'GK': df_sorted[df_sorted['position'] == 'GK'].head(2),
    'DEF': df_sorted[df_sorted['position'] == 'DEF'].head(5),
    'MID': df_sorted[df_sorted['position'] == 'MID'].head(5),
    'FWD': df_sorted[df_sorted['position'] == 'FWD'].head(3),
}
final_team = pd.concat(top_players.values()).drop_duplicates(subset='name')

print("\nFinal Recommended FPL Team:")
print(tabulate(final_team[['name', 'position', 'team', 'Predicted Points', 'value']], headers='keys', tablefmt='grid', showindex=False))

# Save final team CSV to the 'output' folder, overwriting old file
final_team.to_csv(os.path.join(output_dir, "final_recommended_fpl_team.csv"), index=False)
