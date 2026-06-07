import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error

# =========================
# 1. DATASET
# =========================
np.random.seed(42)
n = 1500

experience = np.random.randint(0, 21, n)
education = np.random.randint(1, 5, n)
skills = np.random.randint(1, 11, n)
certifications = np.random.randint(0, 6, n)
age = np.random.randint(20, 60, n)

salary = (
    experience * 2500 +
    education * 5000 +
    skills * 1200 +
    certifications * 2000 +
    age * 100 +
    np.random.normal(0, 3000, n)
)

df = pd.DataFrame({
    "Experience": experience,
    "Education": education,
    "Skills": skills,
    "Certifications": certifications,
    "Age": age,
    "Salary": salary
})

# =========================
# 2. SPLIT DATA
# =========================
X = df.drop("Salary", axis=1)
y = df["Salary"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# 3. SCALING
# =========================
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# =========================
# 4. ANN MODEL
# =========================
model = MLPRegressor(
    hidden_layer_sizes=(64, 32),
    max_iter=2000,
    random_state=42
)

model.fit(X_train, y_train)

# =========================
# 5. EVALUATION
# =========================
pred = model.predict(X_test)
mae = mean_absolute_error(y_test, pred)

print("\n" + "=" * 50)
print("EMPLOYEE SALARY PREDICTION SYSTEM".center(50))
print("=" * 50)

print(f"\nModel Mean Absolute Error: {mae:,.2f}")
print("Model Status: READY\n")
print("=" * 50)

# =========================
# 6. SKILL CONVERTER FUNCTION (NEW)
# =========================
def convert_skills(skill_text):
    skill_list = skill_text.lower().split(",")

    score = 0

    for s in skill_list:
        s = s.strip()

        if "excel" in s:
            score += 3
        elif "python" in s:
            score += 5
        elif "data" in s:
            score += 4
        elif "communication" in s:
            score += 2
        elif "word" in s:
            score += 2
        elif "ai" in s or "ml" in s:
            score += 6
        elif "management" in s:
            score += 4
        else:
            score += 1

    return score

# =========================
# 7. LIVE INPUT SYSTEM
# =========================
def show(title, value):
    print(f"{title:<18}: {value}")

while True:
    try:
        print("\n" + "-" * 50)
        print("ENTER EMPLOYEE DETAILS".center(50))
        print("-" * 50)

        exp = float(input("Experience (years): "))
        edu = int(input("Education (1=Matric,2=Inter,3=Bachelor,4=Master): "))

        # 🔥 NEW SKILLS INPUT (TEXT)
        skill_text = input("Skills (e.g Excel, Python, Communication): ")
        skill_score = convert_skills(skill_text)

        cert = int(input("Certifications: "))
        age = int(input("Age: "))

        new_data = np.array([[exp, edu, skill_score, cert, age]])
        new_data = scaler.transform(new_data)

        salary_pred = model.predict(new_data)[0]

        print("\n" + "-" * 50)
        print("PREDICTION RESULT".center(50))
        print("-"*50)

        show("Experience", f"{exp} years")
        show("Education", edu)
        show("Skills Score", skill_score)
        show("Certifications", cert)
        show("Age", age)

        print("-" * 50)
        print(f"Estimated Salary  : {salary_pred:,.0f}")
        print("-" * 50)

        again = input("\nAnother prediction? (y/n): ")
        if again.lower() != 'y':
            print("\nSystem Closed. Thank you!")
            break

    except ValueError:
        print("Invalid input! Please enter correct values only.")