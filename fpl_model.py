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

# Reproducibility
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Load data
url = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/2023-24/gws/merged_gw.csv"
df = pd.read_csv(url)

# Filter out players who didn't play any minutes
df = df[df['minutes'] > 0]

# Aggregate full-season stats by player
grouped = df.groupby(['name', 'position', 'team', 'element'], as_index=False).agg({
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
    "ict_index": "sum",
    "bps": "sum",
    "bonus": "sum",
    "creativity": "sum",
    "influence": "sum",
    "threat": "sum",
    "value": "mean"
})

# Drop players with less than 300 minutes
grouped = grouped[grouped['minutes'] >= 300]

# Define features and target
features = [
    "goals_scored", "assists", "clean_sheets", "goals_conceded", "yellow_cards", "minutes",
    "expected_goals", "expected_assists", "expected_goal_involvements", "xP", "starts", 
    "ict_index", "bps", "bonus", "creativity", "influence", "threat", "value"
]
target = "total_points"

# Fill NaNs with 0
grouped[features] = grouped[features].fillna(0)

# Scale features
scaler = StandardScaler()
X = scaler.fit_transform(grouped[features])
y = grouped[target].values
joblib.dump(scaler, "scaler.pkl")

# Split data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=SEED)

# Build model
model = Sequential([
    Input(shape=(X.shape[1],)),
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(1)
])

model.compile(optimizer='adam', loss='huber', metrics=['mae'])

callbacks = [
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1)
]

# Train model
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=32,
    verbose=2,
    callbacks=callbacks
)

# Evaluate
val_loss, val_mae = model.evaluate(X_val, y_val, verbose=0)
print(f"\nValidation MAE: {val_mae:.2f}")

# Predict and add to dataframe
y_pred = model.predict(X).flatten()
grouped['Predicted Points'] = y_pred

# Format value
grouped['value'] = (grouped['value'] / 10).round(1)

# Sort output
df_sorted = grouped[['name', 'Predicted Points', 'position', 'team', 'value', 'total_points']].sort_values(by='Predicted Points', ascending=False)

# Create output directory
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

# Position-specific outputs with suffix _Mdl_2024_25
positions = ['GK', 'DEF', 'MID', 'FWD']
for position in positions:
    position_players = df_sorted[df_sorted['position'] == position].drop_duplicates(subset='name')
    filename = f"{position}_players_Mdl_2024_25.csv"
    position_players.to_csv(os.path.join(output_dir, filename), index=False)
    
    print(f"\nRecommended {position}s:")
    print(tabulate(position_players[['name', 'position', 'team', 'Predicted Points', 'value']], headers='keys', tablefmt='grid', showindex=False))
    print("\n" + "-"*50)

# Final team with filename Team_mdl_2024_25.csv
top_players = {
    'GK': df_sorted[df_sorted['position'] == 'GK'].head(2),
    'DEF': df_sorted[df_sorted['position'] == 'DEF'].head(5),
    'MID': df_sorted[df_sorted['position'] == 'MID'].head(5),
    'FWD': df_sorted[df_sorted['position'] == 'FWD'].head(3),
}
final_team = pd.concat(top_players.values()).drop_duplicates(subset='name')

print("\nFinal Recommended FPL Team:")
print(tabulate(final_team[['name', 'position', 'team', 'Predicted Points', 'value']], headers='keys', tablefmt='grid', showindex=False))

final_team.to_csv(os.path.join(output_dir, "Team_mdl_2024_25.csv"), index=False)

