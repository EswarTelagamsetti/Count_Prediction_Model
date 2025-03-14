import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Load the dataset
data = pd.read_csv("refined_prawn_aquaculture_data.csv")

# Encode categorical features
label_encoder = LabelEncoder()
data['Season'] = label_encoder.fit_transform(data['Season'])

# Split features and labels
X = data[['Age_of_Pond', 'Food_Intake', 'Season']]
y = data['Prawn_Count']

# Convert target variable to numeric if needed
y = pd.to_numeric(y, errors='coerce').fillna(0)  # Handling non-numeric values

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print(f"Model MAE: {mae:.2f}")

# Save the trained model and label encoder
with open("prawn_model.pkl", "wb") as f:
    pickle.dump((model, label_encoder), f)