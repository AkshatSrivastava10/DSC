import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv('student_extended_ml_dataset2.csv')

# Univariate Analysis
# Visualize the distribution of the target variable (e.g., physics marks)
plt.figure(figsize=(8, 5))
sns.histplot(df['physics_marks'], bins=20, kde=True)
plt.xlabel('Physics Marks')
plt.ylabel('Frequency')
plt.title('Distribution of Physics Marks')
plt.show()

# Bivariate Analysis
# Explore the correlation between study hours and physics marks
plt.figure(figsize=(8, 5))
sns.scatterplot(x='study_hours', y='physics_marks', data=df)
plt.xlabel('Study Hours')
plt.ylabel('Physics Marks')
plt.title('Relationship Between Study Hours and Physics Marks')
plt.show()

# Multivariate Analysis
# Explore correlations between multiple variables using a heatmap
correlation_matrix = df.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

# Data Preprocessing
# Encode categorical variables (e.g., gender, part-time job)
df_encoded = pd.get_dummies(df, columns=['gender', 'has_part_time_job'])

# Split data into train and test sets
X = df_encoded.drop(['name', 'physics_marks'], axis=1)
y = df_encoded['physics_marks']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Building (Linear Regression)
model = LinearRegression()
model.fit(X_train, y_train)

# Model Evaluation
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')

# Insights
# Coefficients of the linear regression model
coefficients = pd.DataFrame({'Feature': X.columns, 'Coefficient': model.coef_})
coefficients = coefficients.sort_values(by='Coefficient', ascending=False)
print(coefficients)