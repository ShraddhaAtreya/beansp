import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

class FacultyData:
    def __init__(self, num_samples=100):
        self.num_samples = num_samples
        self.df = self.generate_data()

    def generate_data(self):
        data = []
        for _ in range(self.num_samples):
            experience = random.randint(1, 35)
            age = experience + random.randint(25, 30)

            if experience < 10:
                designation, factor = "Asst", 3
            elif 10 <= experience < 20:
                designation, factor = "Asso", 2
            else:
                designation, factor = "Prof", 1

            data.append([experience, age, designation, factor])

        df = pd.DataFrame(data, columns=["Experience", "Age", "Designation", "Designation_Factor"])
        return df

    def multiple_logistic_regression(self):
        X = self.df[["Experience", "Age"]]  # Multiple predictors
        y = self.df["Designation_Factor"]

        model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=500)
        model.fit(X, y)

        print("\nMultiple Logistic Regression:")
        print(f"Coefficients: {model.coef_}")
        print(f"Intercept: {model.intercept_}")
        print(f"Label Mapping (Manual): {{'Prof': 1, 'Asso': 2, 'Asst': 3}}")

        plt.figure(figsize=(8, 5))
        plt.scatter(self.df["Experience"], y, color='blue', label='Experience vs Designation')
        plt.scatter(self.df["Age"], y, color='green', label='Age vs Designation')

        plt.xlabel("Experience & Age")
        plt.ylabel("Designation Factor")
        plt.title("Multiple Logistic Regression: Experience & Age vs Designation")
        plt.legend()
        plt.show()

faculty_data = FacultyData(num_samples=100)
faculty_data.multiple_logistic_regression()
