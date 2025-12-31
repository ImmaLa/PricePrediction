# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


#Load the Dataset

data = pd.read_csv('house_price_prediction_dataset.csv')

# Display the first few rows
data.head(10)

data.tail(10)

data.info()

# descriptive statistics
data.describe()

# Remove irrelevant features

data.drop("PropertyID", axis=1, inplace=True)

# Handle Missing Values
# Check for missing values
data.isnull().mean()*100

# Impute missing values with the median for numerical columns
data['SizeInSqFt'] = data['SizeInSqFt'].fillna(data['SizeInSqFt'].median())
data['Bedrooms'] = data['Bedrooms'].fillna(data['Bedrooms'].median())
data['Bathrooms'] = data['Bathrooms'].fillna(data['Bathrooms'].median())
data['LotSize'] = data['LotSize'].fillna(data['LotSize'].median())

# Encode Categorical Variables
# Convert categorical variables into numerical formats.

from sklearn.preprocessing import OneHotEncoder, LabelEncoder

encoder = LabelEncoder()
data["Location"] = encoder.fit_transform(data["Location"])
data["PropertyType"] = encoder.fit_transform(data["PropertyType"])

# Viewing encoded data
data.head()

#Feature Scaling
# Scale numerical features to standardize their range.
#Import scaler
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

# Scale numerical columns
scaler = MinMaxScaler()
scaled_cols = ['SizeInSqFt', 'Bedrooms', 'Bathrooms', 'YearBuilt', 'GarageSpaces', 'LotSize', 'NearbySchools']
data[scaled_cols] = scaler.fit_transform(data[scaled_cols])

# Display scaled data
data.describe()

# Define features (X) and target variable (y)
X = data.drop('SellingPrice', axis=1)
y = data['SellingPrice']

# Data Splitting
# Split the data into training and testing sets (80% training, 20% testing).
from sklearn.model_selection import train_test_split

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Selection and Training
# Starting with a basic linear regression model.
#Train Linear Regression Model
from sklearn.linear_model import LinearRegression


# Initialize and train the model
lr = LinearRegression()
lr.fit(X_train, y_train)


# Coefficients and intercept
print("Coefficients:", lr.coef_)
print("Intercept:", lr.intercept_)


# Predict on the Test Set
# Predict selling prices on the test set
y_pred = lr.predict(X_test)

# Model Evaluation

# Metrics for Evaluation
# Evaluate the model using common regression metrics:
# - Mean Absolute Error (MAE)
# - Root Mean Squared Error (RMSE)
# - R² Score


from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# Calculate metrics
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("Mean Absolute Error (MAE):", mae)
print("Root Mean Squared Error (RMSE):", rmse)
print("R² Score:", r2)


# Residual Analysis
# Analyze residuals to assess the model's performance.

# Plot residuals
residuals = y_test - y_pred
plt.figure(figsize=(18, 6))
sns.histplot(residuals, kde=True, bins=30)
plt.title("Residual Distribution")
plt.xlabel("Residuals")
plt.ylabel("Frequency")
plt.show()

# Create a DataFrame of feature importance
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': lr.coef_
}).sort_values(by='Coefficient', ascending=False)

