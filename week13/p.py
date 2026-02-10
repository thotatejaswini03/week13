import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

st.title("Campus Placement Prediction System")

# Load dataset
df = pd.read_csv("placement.csv")

st.subheader("Raw Dataset")
st.write(df.head())

# ---------------- DATA CLEANING ----------------

# Drop ID column
df = df.drop("Student_ID", axis=1)

# Remove duplicates
df = df.drop_duplicates()

# Handle missing values
df = df.fillna(df.mean(numeric_only=True))

# Encoding categorical columns
le_gender = LabelEncoder()
le_degree = LabelEncoder()
le_branch = LabelEncoder()
le_target = LabelEncoder()

df["Gender"] = le_gender.fit_transform(df["Gender"])
df["Degree"] = le_degree.fit_transform(df["Degree"])
df["Branch"] = le_branch.fit_transform(df["Branch"])
df["Placement_Status"] = le_target.fit_transform(df["Placement_Status"])

st.subheader("Cleaned Dataset")
st.write(df.head())

# ---------------- MODEL TRAINING ----------------

X = df.drop("Placement_Status", axis=1)
y = df["Placement_Status"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# ---------------- EVALUATION ----------------

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

st.subheader("Model Performance")
st.write("Accuracy:", round(acc*100,2), "%")

st.subheader("Confusion Matrix (Plot)")

cm = confusion_matrix(y_test, y_pred)

fig = plt.figure()
plt.imshow(cm)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")

for i in range(len(cm)):
    for j in range(len(cm[0])):
        plt.text(j, i, cm[i][j], ha='center', va='center')

st.pyplot(fig)


# Feature importance
st.subheader("Feature Importance")
importances = model.feature_importances_
features = X.columns

fig = plt.figure()
plt.barh(features, importances)
plt.xlabel("Importance")
plt.ylabel("Features")
st.pyplot(fig)

# ---------------- PREDICTION UI ----------------

st.header("Enter Student Details")

age = st.number_input("Age", 17, 30)
gender = st.selectbox("Gender", le_gender.classes_)
degree = st.selectbox("Degree", le_degree.classes_)
branch = st.selectbox("Branch", le_branch.classes_)
cgpa = st.number_input("CGPA", 0.0, 10.0, step=0.1)
internships = st.number_input("Internships", 0, 5)
projects = st.number_input("Projects", 0, 10)
coding = st.slider("Coding Skills", 1, 10)
communication = st.slider("Communication Skills", 1, 10)
aptitude = st.slider("Aptitude Score", 0, 100)
soft = st.slider("Soft Skills Rating", 1, 10)
certifications = st.number_input("Certifications", 0, 10)
backlogs = st.number_input("Backlogs", 0, 10)

gender_val = le_gender.transform([gender])[0]
degree_val = le_degree.transform([degree])[0]
branch_val = le_branch.transform([branch])[0]

if st.button("Predict Placement"):

    input_data = [[age, gender_val, degree_val, branch_val, cgpa,
                   internships, projects, coding, communication,
                   aptitude, soft, certifications, backlogs]]

    prediction = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][prediction]

    result = le_target.inverse_transform([prediction])[0]

    st.success(f"Prediction: {result}")
    st.write(f"Confidence: {round(prob*100,2)} %")