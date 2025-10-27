import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

# 1. Load and preprocess dataset
df = pd.read_csv('Stress.csv')

target = 'Depression'
reduced_features = ['Age', 'Gender', 'Work Pressure', 'Sleep Duration', 'Financial Stress']

# Map sleep duration text to numbers
sleep_map = {
    "Less than 4 hours": 2,
    "4-6 hours": 5,
    "6-8 hours": 7,
    "More than 8 hours": 9
}
df['Sleep Duration'] = df['Sleep Duration'].map(sleep_map).fillna(7)

# Label encode categorical columns
label_encoders = {}
for col in ['Gender']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

X = df[reduced_features]
y = df[target]

X = X.fillna(X.mean())

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# 2. Define models & parameters (Simplified for faster training)
models = {
    'Random Forest': (RandomForestClassifier(random_state=42), {
        'n_estimators': [100],
        'max_depth': [10],
    }),
    'SVM': (SVC(probability=True), {
        'C': [1],
        'kernel': ['linear'],
    }),
    'XGBoost': (XGBClassifier(use_label_encoder=False, eval_metric='logloss'), {
        'n_estimators': [100],
        'max_depth': [5],
        'learning_rate': [0.1],
    })
}

# 3. Train with GridSearchCV
results = []
best_model = None
best_accuracy = 0

for name, (model, params) in models.items():
    print(f"\n🔍 Training {name} with GridSearchCV...")
    grid = GridSearchCV(model, params, cv=3, scoring='accuracy', n_jobs=-1)
    grid.fit(X_train, y_train)
    
    best_estimator = grid.best_estimator_
    y_pred = best_estimator.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted')
    rec = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    results.append({
        'Model': name,
        'Best Params': grid.best_params_,
        'Accuracy': round(acc, 3),
        'Precision': round(prec, 3),
        'Recall': round(rec, 3),
        'F1 Score': round(f1, 3)
    })
    
    # Track best model
    if acc > best_accuracy:
        best_accuracy = acc
        best_model = best_estimator
        best_model_name = name

# 4. Compare results
results_df = pd.DataFrame(results)
print("\n📊 Model Comparison:")
print(results_df)

# Save comparison table
results_df.to_csv('model_comparison.csv', index=False)

# 5. Save the best model
print(f"\n🏆 Best Model: {best_model_name}")

# Save everything
joblib.dump(best_model, 'stress_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(label_encoders, 'label_encoders.pkl')
joblib.dump(reduced_features, 'feature_names.pkl')

print("\n✅ Model training, comparison, and saving complete!")

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# -----------------------------
# 6. Visualization - Model Accuracy Comparison
# -----------------------------
plt.figure(figsize=(8, 5))
sns.barplot(x='Model', y='Accuracy', data=results_df, palette=['#FFA500', '#87CEEB', '#32CD32'])
plt.title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
plt.xlabel('Models')
plt.ylabel('Accuracy')
plt.ylim(0, 1)

# Add accuracy values above bars
for i, v in enumerate(results_df['Accuracy']):
    plt.text(i, v + 0.01, str(v), ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig('model_accuracy_comparison.png')
plt.show()

# -----------------------------
# 7. Confusion Matrix for Each Model
# -----------------------------
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('Confusion Matrices for Models', fontsize=14, fontweight='bold')

for ax, (name, (model, params)) in zip(axes, models.items()):
    # Retrain best model for confusion matrix
    grid = GridSearchCV(model, params, cv=3, scoring='accuracy', n_jobs=-1)
    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_title(name)
    ax.set_xlabel('Predicted label')
    ax.set_ylabel('True label')

plt.tight_layout()
plt.savefig('confusion_matrices.png')
plt.show()

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# -----------------------------
# 6. Plot confusion matrices
# -----------------------------
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("Confusion Matrices for Models", fontsize=14, fontweight='bold')

for ax, (name, (model, _)) in zip(axes, models.items()):
    # Retrain model with best params for fair comparison
    grid = GridSearchCV(model, _, cv=3, scoring='accuracy', n_jobs=-1)
    grid.fit(X_train, y_train)
    y_pred = grid.best_estimator_.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax)
    ax.set_title(name)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")

plt.tight_layout()
plt.savefig("confusion_matrices.png", dpi=300)
plt.show()


