# AI-Based Healthcare Advisor System

## Project Overview
The AI-Based Healthcare Advisor System is a Python and Streamlit application designed to provide intelligent clinical decision support. It combines patient registration, medical history management, symptom analysis, drug recommendation, risk assessment, and drug interaction detection into a user-friendly web interface.

The system uses:
- **Streamlit** for the interactive frontend
- **SQLite** for lightweight storage of users, patient entries, and prescriptions
- **Scikit-learn** for machine learning disease prediction
- **Pandas** for dataset processing and analytics

## Key Features
- Patient registration and login
- Secure storage of user data in SQLite
- Patient medical data entry with clearly labeled fields
- AI-powered disease prediction and treatment recommendation
- Drug interaction checking using a dedicated interaction dataset
- Prescription history and risk-level visualization
- Modern web-style layout and visual presentation

## Architecture
1. **Frontend**: `app.py`
   - Uses Streamlit pages and styled containers to create an app-like dashboard
   - Sidebar navigation for page switching
   - Forms for registration, login, and patient medical detail entry
   - Visual cards for analytics, recommendations, and history

2. **Database**: `database.py`
   - Defines `users`, `patients`, and `prescriptions` tables
   - Supports relational consistency via foreign keys

3. **Machine Learning**: `model.py`
   - Trains a Random Forest classifier on healthcare data
   - Encodes categorical fields with label encoders
   - Splits data into training and test sets
   - Evaluates model accuracy and classification performance
   - Saves the trained model and encoders as `.pkl` files

4. **Utilities**: `utils.py`
   - Provides drug interaction lookup
   - Helps the app validate treatment recommendations

## Data Sources
- `healthcare_main_dataset_900.csv`: Contains patient health records, diagnosis labels, and recommended treatment metadata
- `drug_interaction_dataset_900.csv`: Contains known drug interaction pairs with severity labels

## Implementation Details
### Frontend Logic (`app.py`)
- Application starts with a styled landing hero and sidebar navigation
- Registration and login tabs include input validation and secure session state handling
- Patient data entry uses a two-column responsive form for better alignment
- Symptom analysis retrieves the latest patient entry and runs disease prediction
- Interaction results and risk summaries are displayed in visually separated cards
- History page shows patient records and a risk-level bar chart

### Database (`database.py`)
- Creates tables only if they do not already exist
- Uses unique constraint on `users.email`
- Includes foreign keys linking patients back to user accounts
- Stores both raw patient data and prediction results for auditability

### Machine Learning (`model.py`)
- Label-encodes categories: gender, allergies, past diseases, symptoms, and drugs
- Transforms `blood_pressure` into `systolic` and `diastolic`
- Trains a `RandomForestClassifier` with 100 trees
- Evaluates performance on a held-out test set
- Saves model artifacts for use in the runtime app

### Drug Interaction (`utils.py`)
- Loads interaction dataset into a Pandas DataFrame
- Checks both drug orderings for pair matches
- Returns interaction description and severity

## Security and Maintenance
### Security Practices
- Input validation is performed for blood pressure format and required fields
- Use of parameterized SQL statements in the app avoids SQL injection vulnerability
- Only necessary data is stored in the database
- UI uses consistent contrast and accessible font sizes

### Maintainability Practices
- Cleanly separated modules for app logic, model training, database initialization, and utilities
- `requirements.txt` lists dependencies to support reproducible installs
- `README.md` documents architecture, setup, and evaluation
- `.gitignore` excludes generated files and database artifacts
- Model artifacts are persisted, supporting retraining and versioning

### Important Note
- Passwords are currently stored in plain text for simplicity. In a production deployment, implement password hashing (e.g. `bcrypt`) and secure credential handling.

## Model Evaluation
The model was trained and evaluated with a held-out test split.

- **Accuracy:** `0.8944`
- **Precision / Recall / F1-score:**
  - Asthma: 0.96 / 1.00 / 0.98
  - Cold: 0.90 / 0.90 / 0.90
  - Diabetes: 1.00 / 1.00 / 1.00
  - Fever: 0.91 / 0.95 / 0.93
  - Heart Disease: 0.76 / 0.62 / 0.68
  - Hypertension: 0.88 / 1.00 / 0.94
  - Kidney Disease: 0.76 / 0.65 / 0.70
  - Migraine: 0.89 / 0.96 / 0.92

- **Confusion matrix** (rows = true labels, columns = predictions):

```
[[27  0  0  0  0  0  0  0]
 [ 1 18  0  0  0  0  0  1]
 [ 0  0 24  0  0  0  0  0]
 [ 0  0  0 20  1  0  0  0]
 [ 0  1  0  0 13  3  3  1]
 [ 0  0  0  0  0 22  0  0]
 [ 0  1  0  2  3  0 13  1]
 [ 0  0  0  0  0  0  1 24]]
```

## Setup Instructions
1. Create a Python environment (recommended):
   ```bash
   python -m venv venv
   .\venv\Scripts\activate
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Train the model (generates model artifacts):
   ```bash
   python model.py
   ```
4. Start the app:
   ```bash
   streamlit run app.py
   ```

## File Structure
- `app.py` — Streamlit frontend and runtime application
- `database.py` — SQLite schema creation and initialization
- `model.py` — Model training, evaluation, and artifact export
- `utils.py` — Drug interaction helper
- `requirements.txt` — Python dependency manifest
- `README.md` — Project documentation
- `healthcare_main_dataset_900.csv` — Training dataset
- `drug_interaction_dataset_900.csv` — Drug interaction dataset
- `healthcare.db` — Generated SQLite database
- `disease_model.pkl`, `le_*.pkl` — Saved model artifacts

## Future Enhancements
- Add password hashing and user session expiration
- Replace SQLite with a production-grade database like PostgreSQL
- Add dedicated logging and monitoring for runtime errors
- Improve ML model with richer feature engineering and cross-validation
- Add unit tests and integration tests for frontend and backend flows
