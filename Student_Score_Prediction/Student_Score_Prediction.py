import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df = pd.read_csv("StudentPerformanceFactors.csv")
# print(df.head())
# print(df.info())

df = df.dropna()
df = df.drop_duplicates()

plt.scatter(df['Hours_Studied'], df['Exam_Score'], color ='blue')
plt.xlabel(df['Hours_Studied'])
plt.ylabel(df['Exam_Score'])
plt.title("Exam Score vs Study Hours")
#plt.show()
print(df[['Hours_Studied','Exam_Score']].corr())

X = df[['Hours_Studied']]
y = df['Exam_Score']

X_train , X_test, y_train, y_test = train_test_split(X, y ,test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
print("Coefficient:", model.coef_)
print("Intercept:", model.intercept_)

y_pred = model.predict(X_test)

#Evaluation 
print("MAE:", mean_absolute_error(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R² Score:", r2_score(y_test, y_pred))

# Visualiztion of predictions 
plt.scatter(X_test, y_test, color='blue', label="Actual")
plt.plot(X_test, y_pred, color='red', linewidth=2, label="Predicted")
plt.xlabel("Study Hours")
plt.ylabel("Exam Score")
plt.title("Linear Regression: Study Hours vs Exam Score")
plt.legend()
plt.show()
