import sqlite3

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

print("Database Created Successfully")