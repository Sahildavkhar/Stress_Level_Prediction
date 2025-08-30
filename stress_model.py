import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load dataset
df = pd.read_csv('Stress.csv')

# Define target and features (reduced set)
target = 'Depression'  # Change to your actual target column if different
reduced_features = ['Age', 'Gender', 'Academic Pressure', 'Sleep Duration', 'Financial Stress']

# Map Sleep Duration string values to numeric
sleep_map = {
    "Less than 4 hours": 2,
    "4-6 hours": 5,
    "6-8 hours": 7,
    "More than 8 hours": 9
}
df['Sleep Duration'] = df['Sleep Duration'].map(sleep_map).fillna(7)  # fill missing with 7

# Encode categorical features
label_encoders = {}

# Encode 'Gender' column (you can add more categorical columns if needed)
for col in ['Gender']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

# Select features and target
X = df[reduced_features]
y = df[target]

# Handle any missing values if necessary (example: fillna)
X = X.fillna(X.mean())  # Or choose appropriate method

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Save model, scaler, label encoders, and feature names for app use
joblib.dump(model, 'stress_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(label_encoders, 'label_encoders.pkl')
joblib.dump(reduced_features, 'feature_names.pkl')

print("Model training and saving complete.")
