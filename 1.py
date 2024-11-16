import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Load the dataset from CSV
df = pd.read_csv('house_data.csv')

# Check the data types of the columns
print(df.dtypes)

# Convert date columns to datetime if necessary
# Uncomment and modify if you have a date column
# df['date_column'] = pd.to_datetime(df['date_column'])

# Drop non-numeric columns for correlation analysis
df_numeric = df.select_dtypes(include=['int64', 'float64'])

# Exploratory Data Analysis (EDA)
print(df.head())
print(df.describe())
print(df.isnull().sum())

# Correlation matrix
correlation_matrix = df_numeric.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()

# Preprocessing
X = df[['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'waterfront', 'view', 'condition']]
y = df['price']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
print(f"R² Score: {r2}")

# Visualize predictions
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.plot([y.min(), y.max()], [y.min(), y.max()], color='red', linestyle='--')
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual vs Predicted Prices')
plt.show()

# Feature importance
coefficients = pd.DataFrame(model.coef_, X.columns, columns=['Coefficient'])
print(coefficients)

# Save the model
joblib.dump(model, 'house_price_model.pkl')

# Predictions and Visualization
# Residual plot to check the model's performance
residuals = y_test - y_pred
plt.figure(figsize=(10, 6))
plt.scatter(y_test, residuals)
plt.axhline(y=0, color='red', linestyle='--')
plt.xlabel("Actual Prices")
plt.ylabel("Residuals")
plt.title("Residual Plot")
plt.show()

# Use the trained model to make predictions on new data
new_data = pd.DataFrame([[3, 2, 1500, 4000, 1, 0, 0, 3]], 
                         columns=['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 
                                  'floors', 'waterfront', 'view', 'condition'])  # Ensure it has the same columns

predicted_price = model.predict(new_data)

print("Predicted Price:", predicted_price[0])