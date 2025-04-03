import numpy as np
import random
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Lambda functions for faculty data generation
assign_designation = lambda exp: "Professor" if exp >= 18 else "Associate Professor" if exp >= 11 else "Assistant Professor"
designation_to_factor = lambda desig: 1 if desig == "Professor" else 2 if desig == "Associate Professor" else 3
calculate_salary = lambda desig_factor: random.randint(105000, 145000) if desig_factor == 1 else random.randint(67000, 100000) if desig_factor == 2 else random.randint(40000, 65000)
generate_patents = lambda pub: max(0, random.randint(0, pub // 2))

# Create faculty data
num_faculty = 100
faculty_data = []

for i in range(num_faculty):
    experience = random.randint(1, 35)
    designation = assign_designation(experience)
    designation_factor = designation_to_factor(designation)
    publications = random.randint(1, 50)
    
    faculty_data.append({
        "name": f"Faculty{i+1}",
        "faculty_id": f"ID{i+1:03d}",
        "experience": experience,
        "gender": random.choice(['Male', 'Female']),
        "designation": designation,
        "designation_factor": designation_factor,
        "salary": calculate_salary(designation_factor),
        "publications": publications,
        "patents": generate_patents(publications)
    })

# Create DataFrame
df = pd.DataFrame(faculty_data)

# Display sample data
print("\nSample Dataset:\n", df.head())

# Correlation matrix
correlation_matrix = df[['designation_factor', 'salary', 'experience', 'publications', 'patents']].corr()
print("\nCorrelation Matrix:\n", correlation_matrix)

# ===================== Regression Model 1: Predict Salary =====================
# Multiple regression: Salary ~ Designation Factor + Experience
X_salary = df[['designation_factor', 'experience']]
y_salary = df['salary']
salary_model = LinearRegression().fit(X_salary, y_salary)
df['predicted_salary'] = salary_model.predict(X_salary)

# Visualizing Salary Prediction
plt.figure(figsize=(10, 6))
plt.scatter(df['experience'], df['salary'], c=df['designation_factor'], cmap='viridis', alpha=0.6, label="Actual Salary")
plt.scatter(df['experience'], df['predicted_salary'], color='red', alpha=0.6, label="Predicted Salary")
plt.xlabel("Experience (Years)")
plt.ylabel("Salary")
plt.title("Actual vs Predicted Salary by Experience and Designation")
plt.colorbar(label="Designation Factor")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# ===================== Regression Model 2: Predict Patents =====================
# Multiple regression: Patents ~ Publications + Experience
X_patents = df[['publications', 'experience']]
y_patents = df['patents']
patents_model = LinearRegression().fit(X_patents, y_patents)
df['predicted_patents'] = patents_model.predict(X_patents)

# Visualizing Patents Prediction
plt.figure(figsize=(10, 6))
plt.scatter(df['publications'], df['patents'], c=df['experience'], cmap='plasma',alpha=0.6, label="Actual Patents")
plt.scatter(df['publications'], df['predicted_patents'], color='orange', alpha=0.6, label="Predicted Patents")
plt.xlabel("Publications")
plt.ylabel("Patents")
plt.title("Actual vs Predicted Patents by Publications and Experience")
plt.colorbar(label="Experience (Years)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
