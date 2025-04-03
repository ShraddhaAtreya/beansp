import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from mpl_toolkits.mplot3d import Axes3D

def generate_data(n=100):
    names = [f"Faculty_{i}" for i in range(n)]
    ids = [f"ID{i:03d}" for i in range(n)]
    experience = np.random.randint(1, 35, n)
    departments = ['CSE', 'ECE', 'EEE', 'MECH', 'CIVIL']
    designations = ['Assistant Professor', 'Associate Professor', 'Professor']
    salaries = np.random.randint(40000, 150000, n)
    genders = random.choices(['Male', 'Female'], k=n)
    publications = np.random.randint(0, 30, n)
    patents = np.random.randint(0, 10, n)
    ages = np.random.randint(28, 65, n)

    data = {
        'Name': names,
        'ID': ids,
        'Experience': experience,
        'Dept': random.choices(departments, k=n),
        'Designation': random.choices(designations, k=n),
        'Salary': salaries,
        'Gender': genders,
        'Publications': publications,
        'Patents': patents,
        'Age': ages
    }

    return pd.DataFrame(data)

# ----------------------------
# Create Data
# ----------------------------
df = generate_data(100)

# ----------------------------
# a) 3D Plot for Publications per Dept
# ----------------------------
fig = plt.figure(figsize=(7,5))
ax = fig.add_subplot(111, projection='3d')

dept_publications = df.groupby('Dept')['Publications'].sum()
x = np.arange(len(dept_publications))
y = np.zeros_like(x)
z = dept_publications.values

ax.bar3d(x, y, np.zeros_like(z), 0.5, 0.5, z, shade=True)

ax.set_xticks(x)

ax.set_xlabel("Department")
ax.set_ylabel("Publication Count")
ax.set_zlabel("Total Publications")
ax.set_title("3D Plot of Publications per Department")

plt.show()

# b) Box Plot for Patents per Dept

plt.figure(figsize=(7,5))
df.boxplot(column='Patents', by='Dept', grid=False)
plt.title("Box Plot of Patents per Department")

plt.xlabel("Department")
plt.ylabel("Number of Patents")
plt.show()


# Stem Plot for Assistant Professors per Dept (Fixed)

plt.figure(figsize=(7,5))
plt.stem(assistant_professors.index, assistant_professors.values)  # Removed use_line_collection=True
plt.title("Stem Plot of Assistant Professors per Department")
plt.xlabel("Department")
plt.ylabel("Number of Assistant Professors")
plt.grid(True, linestyle="--", alpha=0.5)
plt.show()


# ----------------------------
# d) Stack Plot for Age vs Experience
# ----------------------------
plt.figure(figsize=(7,5))
plt.stackplot(df.index, df['Age'], df['Experience'], labels=['Age', 'Experience'], alpha=0.6)
plt.title("Stack Plot of Age and Experience")
plt.xlabel("Faculty Index")
plt.ylabel("Value")
plt.legend()
plt.show()
