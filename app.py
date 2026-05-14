import streamlit as st
import sqlite3
import pandas as pd
import joblib
from utils import check_interaction

# Initialize database
conn = sqlite3.connect("healthcare.db")
cursor = conn.cursor()

# Users table for registration and login
cursor.execute("""
CREATE TABLE IF NOT EXISTS users(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    email TEXT UNIQUE,
    password TEXT,
    age INTEGER,
    gender TEXT
)
""")

# Patients table for medical history
cursor.execute("""
CREATE TABLE IF NOT EXISTS patients(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER,
    blood_pressure TEXT,
    blood_sugar INTEGER,
    allergies TEXT,
    past_diseases TEXT,
    current_symptoms TEXT,
    previous_drugs TEXT,
    diagnosed_disease TEXT,
    recommended_drug TEXT,
    alternative_drug TEXT,
    side_effects TEXT,
    risk_level TEXT,
    FOREIGN KEY(user_id) REFERENCES users(id)
)
""")

# Prescriptions table
cursor.execute("""
CREATE TABLE IF NOT EXISTS prescriptions(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id INTEGER,
    drug TEXT,
    dosage TEXT,
    date_prescribed TEXT,
    FOREIGN KEY(patient_id) REFERENCES patients(id)
)
""")

conn.commit()
conn.close()

# Load model and encoders
model = joblib.load('disease_model.pkl')
le_gender = joblib.load('le_gender.pkl')
le_allergies = joblib.load('le_allergies.pkl')
le_past_diseases = joblib.load('le_past_diseases.pkl')
le_current_symptoms = joblib.load('le_current_symptoms.pkl')
le_previous_drugs = joblib.load('le_previous_drugs.pkl')
le_diagnosed_disease = joblib.load('le_diagnosed_disease.pkl')


def safe_transform(encoder, value):
    if value is None:
        value = 'None'
    value = str(value).strip()
    if value == '':
        value = 'None'
    if value in encoder.classes_:
        return encoder.transform([value])[0]
    return 0

# Load datasets
healthcare_df = pd.read_csv('healthcare_main_dataset_900.csv')

st.set_page_config(page_title="AI Healthcare Advisor", page_icon="🩺", layout="wide")

st.markdown(
    """
    <style>
        .stApp {
            background: linear-gradient(135deg, #f8fafc 0%, #ffffff 100%);
            color: #0f172a;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            font-size: 16px;
        }
        .hero-card, .feature-card, .result-card, .history-card {
            border-radius: 22px;
            background: rgba(255,255,255,0.95);
            box-shadow: 0 18px 60px rgba(15, 23, 42, 0.08);
            padding: 20px 24px;
            margin-bottom: 20px;
            border: 1px solid #e2e8f0;
        }
        .hero-title {
            font-size: 3.2rem;
            font-weight: 800;
            margin-bottom: 0.2rem;
            color: #1e293b;
            text-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .hero-subtitle {
            font-size: 1.15rem;
            color: #475569;
            line-height: 1.7;
        }
        .feature-list li {
            margin-bottom: 0.65rem;
            line-height: 1.7;
            color: #334155;
            font-size: 1.05rem;
        }
        .section-heading {
            font-size: 1.9rem;
            margin-bottom: 0.6rem;
            color: #1e293b;
            font-weight: 700;
        }
        .stButton>button {
            background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
            color: white;
            border-radius: 14px;
            padding: 0.9rem 1.5rem;
            font-weight: 600;
            font-size: 1.05rem;
            border: none;
            box-shadow: 0 4px 12px rgba(37, 99, 235, 0.3);
            transition: all 0.3s ease;
        }
        .stButton>button:hover {
            background: linear-gradient(135deg, #1d4ed8 0%, #1e40af 100%);
            box-shadow: 0 6px 16px rgba(37, 99, 235, 0.4);
        }
        .stTextInput>div>div>input,
        .stNumberInput>div>div>input,
        .stSelectbox>div>div>div>div {
            border-radius: 12px;
            border: 2px solid #e2e8f0;
            padding: 0.8rem;
            font-size: 1.05rem;
            color: #334155;
            background-color: #ffffff;
        }
        .stTextInput>div>div>input:focus,
        .stNumberInput>div>div>input:focus,
        .stSelectbox>div>div>div>div:focus {
            border-color: #2563eb;
            box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.1);
            background-color: #ffffff;
        }
        .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
            color: #1e293b !important;
            font-weight: 700 !important;
            font-size: 1.8rem !important;
        }
        .stMarkdown h4, .stMarkdown h5, .stMarkdown h6 {
            color: #1e293b !important;
            font-weight: 600 !important;
            font-size: 1.4rem !important;
        }
        .stMarkdown p, .stMarkdown li {
            color: #475569 !important;
            font-size: 1.05rem !important;
            line-height: 1.6 !important;
        }
        .stTextInput label,
        .stNumberInput label,
        .stSelectbox label {
            color: #1e293b !important;
            font-weight: 600 !important;
            font-size: 1.05rem !important;
        }
        .stSuccess {
            background-color: #dcfce7 !important;
            color: #dc2626 !important;
            border: 1px solid #16a34a !important;
        }
        .stError {
            background-color: #fef2f2 !important;
            color: #a16207 !important;
            border: 1px solid #dc2626 !important;
        }
        .stWarning {
            background-color: #fffbeb !important;
            color: #92400e !important;
            border: 1px solid #d97706 !important;
        }
        .stInfo {
            background-color: #eff6ff !important;
            color: #1e40af !important;
            border: 1px solid #3b82f6 !important;
        }
        .stDataFrame {
            border-radius: 12px;
            overflow: hidden;
            font-size: 1rem;
        }
        .stMetric {
            background: rgba(255,255,255,0.9);
            border-radius: 12px;
            padding: 1rem;
            box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        }
        .stMetric label {
            color: #1e293b;
            font-size: 1.1rem;
            font-weight: 600;
        }
        .stMetric .metric-value {
            color: #2563eb;
            font-size: 1.5rem;
            font-weight: 700;
        }
        .stMetric .metric-delta {
            color: #059669;
            font-size: 0.9rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class='hero-card'>
        <div class='hero-title'>AI-Based Healthcare Advisor</div>
        <div class='hero-subtitle'>A smart clinical decision support assistant for patients and doctors. Analyze symptoms, manage medical history, predict conditions, and get drug recommendations with safety checks.</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Session state for login
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'user_id' not in st.session_state:
    st.session_state.user_id = None

# Navigation sidebar
with st.sidebar:
    st.markdown("## Navigation")
    if st.session_state.logged_in:
        page = st.radio("Go to", ["Home", "Enter Patient Data", "Analyze Symptoms", "View History", "Logout"], index=0)
    else:
        page = st.radio("Go to", ["Login/Register"], index=0)
    st.write("---")
    st.markdown("### Tips")
    st.write("Use the left menu to move between dashboard sections.")
    st.write("Enter your medical details first before analyzing symptoms.")

if page == "Login/Register":
    st.markdown("<div class='feature-card' style='padding: 32px;'>\n        <h3 class='section-heading'>Secure healthcare access</h3>\n        <p>Register as a patient and securely log in to manage medical history, prescriptions, and symptom analysis.</p>\n    </div>", unsafe_allow_html=True)

    col1, col2 = st.columns([1.2, 1])
    with col1:
        st.markdown("<div class='result-card' style='padding: 30px;'>\n            <h3 class='section-heading'>Why choose this app?</h3>\n            <ul class='feature-list'>\n                <li>Modern, clean interface with aligned input controls.</li>\n                <li>Fast patient registration and secure login flow.</li>\n                <li>Smart symptom analysis with drug safety checks.</li>\n                <li>Simple health history tracking and visual reports.</li>\n            </ul>\n        </div>", unsafe_allow_html=True)

    with col2:
        tab1, tab2 = st.tabs(["Login", "Register"])
        with tab1:
            st.markdown("<div class='feature-card' style='padding: 28px;'>\n                <h3 class='section-heading'>Login</h3>\n            </div>", unsafe_allow_html=True)
            email = st.text_input("Email", key="login_email")
            password = st.text_input("Password", type="password", key="login_password")
            if st.button("Login"):
                conn = sqlite3.connect("healthcare.db")
                cursor = conn.cursor()
                cursor.execute("SELECT id, name FROM users WHERE email=? AND password=?", (email, password))
                user = cursor.fetchone()
                conn.close()
                if user:
                    st.session_state.logged_in = True
                    st.session_state.user_id = user[0]
                    st.success(f"Welcome back, {user[1]}!")
                    st.experimental_rerun()
                else:
                    st.error("Invalid credentials. Please try again.")
        with tab2:
            st.markdown("<div class='feature-card' style='padding: 28px;'>\n                <h3 class='section-heading'>Register</h3>\n            </div>", unsafe_allow_html=True)
            name = st.text_input("Full Name", key="reg_name")
            email = st.text_input("Email", key="reg_email")
            password = st.text_input("Password", type="password", key="reg_password")
            age = st.number_input("Age", min_value=1, max_value=120, key="reg_age")
            gender = st.selectbox("Gender", ["Male", "Female", "Other"], key="reg_gender")
            if st.button("Register"):
                conn = sqlite3.connect("healthcare.db")
                cursor = conn.cursor()
                try:
                    cursor.execute("INSERT INTO users(name,email,password,age,gender) VALUES(?,?,?,?,?)", (name,email,password,age,gender))
                    conn.commit()
                    st.success("Registration successful! Please log in.")
                except sqlite3.IntegrityError:
                    st.error("Email already exists. Use a different email.")
                conn.close()

elif page == "Home":
    st.markdown("<div class='hero-card' style='padding: 30px;'>\n        <h3 class='section-heading'>Dashboard</h3>\n        <p>Track your registered users, patient records, and AI-driven predictions all from one interface.</p>\n    </div>", unsafe_allow_html=True)
    conn = sqlite3.connect("healthcare.db")
    cursor = conn.cursor()
    total_users = cursor.execute("SELECT COUNT(*) FROM users").fetchone()[0]
    total_patients = cursor.execute("SELECT COUNT(*) FROM patients").fetchone()[0]
    total_predictions = cursor.execute("SELECT COUNT(*) FROM patients WHERE diagnosed_disease IS NOT NULL").fetchone()[0]
    conn.close()

    metric1, metric2, metric3 = st.columns(3)
    metric1.metric("Registered Users", total_users, delta="+5%")
    metric2.metric("Patient Records", total_patients, delta="+12%")
    metric3.metric("Predictions Made", total_predictions, delta="+8%")

    with st.container():
        st.markdown("<div class='feature-card'>\n            <h3 class='section-heading'>Quick actions</h3>\n            <p>Use the sidebar to enter patient data or analyze symptoms for the latest record saved under your account.</p>\n        </div>", unsafe_allow_html=True)

elif page == "Enter Patient Data":
    st.markdown("<div class='hero-card' style='padding: 30px;'>\n        <h3 class='section-heading'>Patient Data Entry</h3>\n        <p>Enter the patient’s vital signs, allergic history, symptoms, and prior medicines. All fields are clearly labeled for easy input.</p>\n    </div>", unsafe_allow_html=True)

    with st.form("patient_form"):
        left, right = st.columns(2)
        with left:
            blood_pressure = st.text_input("Blood Pressure (e.g., 120/80)")
            blood_sugar = st.number_input("Blood Sugar", min_value=0)
            allergies = st.text_input("Allergies (comma separated)")
        with right:
            past_diseases = st.text_input("Past Diseases (comma separated)")
            current_symptoms = st.text_input("Current Symptoms (comma separated)")
            previous_drugs = st.text_input("Previous Drugs (comma separated)")

        submit = st.form_submit_button("Save Data")
        if submit:
            if '/' not in blood_pressure or not all(part.strip().isdigit() for part in blood_pressure.split('/')):
                st.error("Please enter blood pressure as systolic/diastolic, for example 120/80.")
            else:
                conn = sqlite3.connect("healthcare.db")
                cursor = conn.cursor()
                cursor.execute("""
                INSERT INTO patients(user_id, blood_pressure, blood_sugar, allergies, past_diseases, current_symptoms, previous_drugs)
                VALUES(?,?,?,?,?,?,?)
                """, (st.session_state.user_id, blood_pressure, blood_sugar, allergies, past_diseases, current_symptoms, previous_drugs))
                conn.commit()
                conn.close()
                st.success("Patient data saved successfully!")

elif page == "Analyze Symptoms":
    st.markdown("<div class='hero-card' style='padding: 30px;'>\n        <h3 class='section-heading'>Symptom Analysis</h3>\n        <p>Review the latest patient record and get instant AI-driven condition prediction with drug safety recommendations.</p>\n    </div>", unsafe_allow_html=True)
    conn = sqlite3.connect("healthcare.db")
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM patients WHERE user_id=? ORDER BY id DESC LIMIT 1", (st.session_state.user_id,))
    patient = cursor.fetchone()
    conn.close()

    if not patient:
        st.warning("No patient data found. Please enter your medical details first.")
    else:
        predicted_disease = None
        rec = None
        alt = None
        side_effects = None
        risk = None
        interactions = []

        bp_text = str(patient[2]).strip() if patient[2] is not None else ""
        if '/' not in bp_text:
            st.error("Invalid blood pressure format. Please enter it as systolic/diastolic, for example 120/80.")
        else:
            parts = [part.strip() for part in bp_text.split('/')]
            if len(parts) != 2 or not all(part.isdigit() for part in parts):
                st.error("Invalid blood pressure values. Use numeric values like 120/80.")
            else:
                systolic, diastolic = map(int, parts)
                conn = sqlite3.connect("healthcare.db")
                cursor = conn.cursor()
                cursor.execute("SELECT age, gender FROM users WHERE id=?", (st.session_state.user_id,))
                user_info = cursor.fetchone()
                conn.close()

                if not user_info:
                    st.error("Unable to load user profile.")
                else:
                    age, gender = user_info
                    try:
                        features = [
                            age,
                            safe_transform(le_gender, gender),
                            systolic,
                            diastolic,
                            patient[3],
                            safe_transform(le_allergies, patient[4]),
                            safe_transform(le_past_diseases, patient[5]),
                            safe_transform(le_current_symptoms, patient[6]),
                            safe_transform(le_previous_drugs, patient[7])
                        ]
                        prediction = model.predict([features])[0]
                        predicted_disease = le_diagnosed_disease.inverse_transform([prediction])[0]
                    except Exception as err:
                        st.error(f"Unable to encode patient data for prediction: {err}")

        if predicted_disease is not None:
            st.success(f"Predicted Disease: {predicted_disease}")
            disease_rows = healthcare_df[healthcare_df['diagnosed_disease'] == predicted_disease]
            if not disease_rows.empty:
                rec = disease_rows['recommended_drug'].mode()[0]
                alt = disease_rows['alternative_drug'].mode()[0]
                side_effects = disease_rows['side_effects'].mode()[0]
                risk = disease_rows['risk_level'].mode()[0]

                summary_col, safety_col = st.columns([1.1, 0.9])
                with summary_col:
                    st.markdown("<div class='result-card'>\n                        <h3 class='section-heading'>Recommendation</h3>\n                        <p><strong>Recommended Drug:</strong> {}</p>\n                        <p><strong>Alternative Drug:</strong> {}</p>\n                        <p><strong>Side Effects:</strong> {}</p>\n                    </div>".format(rec, alt, side_effects), unsafe_allow_html=True)
                with safety_col:
                    st.markdown("<div class='result-card'>\n                        <h3 class='section-heading'>Risk & Safety</h3>\n                        <p><strong>Risk Level:</strong> {}</p>\n                    </div>".format(risk), unsafe_allow_html=True)
                    prev_drugs = [d.strip() for d in str(patient[7]).split(',') if d.strip()]
                    if prev_drugs:
                        st.markdown("<div class='feature-card' style='padding: 20px;'>\n                            <h4 style='margin-bottom: 10px;'>Interaction checks</h4>\n                        </div>", unsafe_allow_html=True)
                    for drug in prev_drugs:
                        inter, sev = check_interaction(rec, drug)
                        if inter:
                            interactions.append(f"{rec} with {drug}: {inter} (Severity: {sev})")
                        inter, sev = check_interaction(alt, drug)
                        if inter:
                            interactions.append(f"{alt} with {drug}: {inter} (Severity: {sev})")

                if interactions:
                    st.warning("Potential Drug Interactions:")
                    for item in interactions:
                        st.write(f"- {item}")
                else:
                    st.success("No harmful interactions detected.")

                conn = sqlite3.connect("healthcare.db")
                cursor = conn.cursor()
                cursor.execute("""
                UPDATE patients SET diagnosed_disease=?, recommended_drug=?, alternative_drug=?, side_effects=?, risk_level=?
                WHERE id=?
                """, (predicted_disease, rec, alt, side_effects, risk, patient[0]))
                cursor.execute("INSERT INTO prescriptions(patient_id, drug, dosage, date_prescribed) VALUES(?,?,?,?)", (patient[0], rec, "As prescribed", pd.Timestamp.now().strftime("%Y-%m-%d")))
                conn.commit()
                conn.close()
            else:
                st.warning("No recommendation data found for the predicted disease.")

elif page == "View History":
    st.markdown("<div class='hero-card' style='padding: 30px;'>\n        <h3 class='section-heading'>Medical History</h3>\n        <p>Browse records, check past diagnoses, and view risk-level analytics for your saved patient data.</p>\n    </div>", unsafe_allow_html=True)
    conn = sqlite3.connect("healthcare.db")
    df = pd.read_sql_query("SELECT * FROM patients WHERE user_id=?", conn, params=(st.session_state.user_id,))
    conn.close()

    if df.empty:
        st.info("No patient records yet. Enter data first to start tracking history.")
    else:
        stats1, stats2 = st.columns(2)
        stats1.metric("Records", len(df))
        stats2.metric("Unique Diagnoses", df['diagnosed_disease'].nunique())
        with st.expander("View full medical table"):
            st.dataframe(df)
        st.markdown("<div class='history-card'><h3 class='section-heading'>Risk Level Distribution</h3></div>", unsafe_allow_html=True)
        risk_counts = df['risk_level'].value_counts()
        st.bar_chart(risk_counts)

elif page == "Logout":
    st.session_state.logged_in = False
    st.session_state.user_id = None
    st.success("Logged out!")
    st.experimental_rerun()