import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd
import joblib

# Load healthcare dataset
df = pd.read_csv('healthcare_main_dataset_900.csv')

# Preprocess data
le_gender = LabelEncoder()
le_allergies = LabelEncoder()
le_past_diseases = LabelEncoder()
le_current_symptoms = LabelEncoder()
le_previous_drugs = LabelEncoder()
le_diagnosed_disease = LabelEncoder()

df['gender_encoded'] = le_gender.fit_transform(df['gender'])
df['allergies_encoded'] = le_allergies.fit_transform(df['allergies'])
df['past_diseases_encoded'] = le_past_diseases.fit_transform(df['past_diseases'])
df['current_symptoms_encoded'] = le_current_symptoms.fit_transform(df['current_symptoms'])
df['previous_drugs_encoded'] = le_previous_drugs.fit_transform(df['previous_drugs'])

# Prepare features
X = df[['age', 'gender_encoded', 'blood_pressure', 'blood_sugar', 'allergies_encoded', 'past_diseases_encoded', 'current_symptoms_encoded', 'previous_drugs_encoded']].copy()
X['systolic'] = X['blood_pressure'].apply(lambda x: int(x.split('/')[0]))
X['diastolic'] = X['blood_pressure'].apply(lambda x: int(x.split('/')[1]))
X = X.drop('blood_pressure', axis=1)

# Target
y = le_diagnosed_disease.fit_transform(df['diagnosed_disease'])

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=le_diagnosed_disease.classes_)
confusion = confusion_matrix(y_test, y_pred)

# Save model and encoders
joblib.dump(model, 'disease_model.pkl')
joblib.dump(le_gender, 'le_gender.pkl')
joblib.dump(le_allergies, 'le_allergies.pkl')
joblib.dump(le_past_diseases, 'le_past_diseases.pkl')
joblib.dump(le_current_symptoms, 'le_current_symptoms.pkl')
joblib.dump(le_previous_drugs, 'le_previous_drugs.pkl')
joblib.dump(le_diagnosed_disease, 'le_diagnosed_disease.pkl')

print("Model trained and saved.")
print(f"Accuracy: {accuracy:.4f}")
print("Classification report:")
print(report)
print("Confusion matrix:")
print(confusion)