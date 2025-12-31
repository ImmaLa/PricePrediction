#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# # 1. Data Preparation
# ### 1.1 Load the Dataset

# In[2]:


data = pd.read_csv('house_price_prediction_dataset.csv')


# In[3]:


# Display the first few rows
data.head()


# In[4]:


data.info()


# In[ ]:





# In[5]:


data.describe()


# In[ ]:





# ### Remove irrelevant features

# In[6]:


data.drop("PropertyID", axis=1, inplace=True)


# ### 1.2 Handle Missing Values
# Numerical Features: Impute with median (robust to outliers).

# In[7]:


# Check for missing values
data.isnull().mean()*100


# In[ ]:





# In[8]:


# Impute missing values with the median for numerical columns
data['SizeInSqFt'] = data['SizeInSqFt'].fillna(data['SizeInSqFt'].median())
data['Bedrooms'] = data['Bedrooms'].fillna(data['Bedrooms'].median())
data['Bathrooms'] = data['Bathrooms'].fillna(data['Bathrooms'].median())
data['LotSize'] = data['LotSize'].fillna(data['LotSize'].median())


# In[9]:


# Check for missing values
data.isnull().mean()*100


# In[ ]:





# ### 1.3 Encode Categorical Variables
# Convert categorical variables into numerical formats.

# In[10]:


from sklearn.preprocessing import OneHotEncoder, LabelEncoder

encoder = LabelEncoder()
data["Location"] = encoder.fit_transform(data["Location"])
data["PropertyType"] = encoder.fit_transform(data["PropertyType"])


# In[11]:


data.head()


# ### 1.4 Feature Scaling
# Scale numerical features to standardize their range.

# In[12]:


from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

# Scale numerical columns
scaler = MinMaxScaler()
scaled_cols = ['SizeInSqFt', 'Bedrooms', 'Bathrooms', 'YearBuilt', 'GarageSpaces', 'LotSize', 'NearbySchools']
data[scaled_cols] = scaler.fit_transform(data[scaled_cols])


# In[13]:


# Display scaled data
data.describe()


# In[ ]:





# In[ ]:





# ### 1.6 Define Features and Target

# In[14]:


# Define features (X) and target variable (y)
X = data.drop('SellingPrice', axis=1)
y = data['SellingPrice']


# # 2. Data Splitting
# Split the data into training and testing sets (80% training, 20% testing).

# In[15]:


from sklearn.model_selection import train_test_split

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[ ]:





# # 3. Model Selection and Training
# Start with a basic linear regression model.

# ### 3.1 Train Linear Regression Model

# In[16]:


from sklearn.linear_model import LinearRegression


# In[17]:


# Initialize and train the model
lr = LinearRegression()
lr.fit(X_train, y_train)


# In[18]:


# Coefficients and intercept
print("Coefficients:", lr.coef_)
print("Intercept:", lr.intercept_)


# ***Result Interpretation:***
# 
# - **Intercept (280,311.01):**
#     - The intercept represents the predicted base selling price of a property when all features (predictors) are zero.
#     - In this case, it serves as the baseline price before accounting for factors like size, location, and amenities.
#     - Since having "zero" for features like SizeInSqFt or Bedrooms is unrealistic, the intercept mainly provides a starting value for the model predictions.
#  
# 
# - **Coefficients:**
# Each coefficient represents the change in the predicted selling price for a one-unit increase in the respective feature, holding all other features constant.
# 
# ***Feature	Coefficient	Interpretation***
# | **Feature**       | **Coefficient**   | **Interpretation**                                                                                                                                      |
# |-------------------|-------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------|
# | **Location**      | -14,601.62        | A one-unit increase in `Location` results in a decrease of \$14,601.62 in the predicted selling price, indicating that certain locations may reduce the price. |
# | **PropertyType**  | 21,643.42         | A one-unit increase in `PropertyType` corresponds to an increase of \$21,643.42 in the selling price, suggesting that certain property types (e.g., Single Family) increase the price. |
# | **SizeInSqFt**    | 548,093.23        | A one-unit increase in `SizeInSqFt` results in an increase of \$548,093.23 in the predicted price, indicating a strong positive relationship between size and price. |
# | **Bedrooms**      | 164,040.26        | A one-bedroom increase results in an increase of \$164,040.26 in price, showing that more bedrooms contribute to a higher property value. |
# | **Bathrooms**     | 65,965.42         | Each additional bathroom increases the predicted price by \$65,965.42, suggesting that more bathrooms add value to the property. |
# | **YearBuilt**     | 21,140.17         | A one-year increase in `YearBuilt` leads to an increase of \$21,140.17 in price, indicating that newer homes tend to have a higher price. |
# | **GarageSpaces**  | 40,574.41         | Each additional garage space increases the price by \$40,574.41, reflecting the value added by more garage spaces. |
# | **LotSize**       | -401,928.69       | A one-unit increase in `LotSize` results in a decrease of \$401,928.69 in price, possibly indicating that larger lot sizes in certain areas reduce the property value. |
# | **NearbySchools** | 49,468.95         | A one-unit increase in `NearbySchools` leads to an increase of \$49,468.95 in the predicted price, indicating that properties near highly rated schools are valued higher. |
# | **MarketTrend**   | 11,909.85         | A one-unit increase in `MarketTrend` corresponds to an increase of \$11,909.85 in price, suggesting that properties in areas with improving market trends are valued higher. |
# 
#                        
# 
# - **Insights:**
#     1. High Impact Features on Property Price:
#         - SizeInSqFt and Bedrooms have the highest coefficients, suggesting that the size of the property and the number of bedrooms are crucial factors in determining the property’s selling price. Larger properties with more bedrooms are significantly more valuable.
# 
#     2. Positive Correlation with Price:
# 
#         - Features such as PropertyType, Bathrooms, YearBuilt, GarageSpaces, NearbySchools, and MarketTrend all show positive coefficients, meaning that increases in these factors generally lead to higher property prices. For example, being closer to good schools or having more bathrooms or garage spaces tends to increase the value of a property.
# 
#     3. Negative Impact of Certain Features:
# 
#         - Location and LotSize have negative coefficients, meaning that properties in certain locations (perhaps less desirable areas) or with larger lot sizes may have a reduced selling price. Larger lot sizes, for instance, might not always translate to higher value, possibly due to lower demand for large land areas or specific geographic areas.
# 
#     4. Significance of Market and Location:
# 
#         - MarketTrend and Location both show how external factors influence the price. A better market trend increases property value, while a less desirable location reduces it.
# 
#     5. Subtle Influence of YearBuilt:
# 
#         - The YearBuilt feature also contributes positively, but with a relatively smaller impact compared to others like size or bedrooms. Newer properties tend to be worth more, but this relationship is not as pronounced as other variables.

# ### 3.2 Predict on the Test Set

# In[20]:


# Predict selling prices on the test set
y_pred = lr.predict(X_test)


# In[ ]:





# # 4. Model Evaluation

# ### 4.1 Metrics for Evaluation
# Evaluate the model using common regression metrics:
# - Mean Absolute Error (MAE)
# - Root Mean Squared Error (RMSE)
# - R² Score

# In[21]:


from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# Calculate metrics
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("Mean Absolute Error (MAE):", mae)
print("Root Mean Squared Error (RMSE):", rmse)
print("R² Score:", r2)


# **1. Mean Absolute Error (MAE): \$67,620.67**
# - On average, the model's predictions deviate from the actual house prices by \$67,620.
# - This value provides a direct measure of prediction error in dollars, making it easy for stakeholders to understand.
#   
# **2. Root Mean Squared Error (RMSE): \$90,705.03**
# - RMSE penalizes larger errors more than MAE due to squaring differences.
# - An RMSE of ~$90,705 suggests the model has some room for improvement, especially for outlier predictions.
#     
# **3. R² (R-Squared): 0.86**
# - The model explains 86% of the variance in house prices based on the given features.
# - This is a strong result, indicating that the model captures most of the important factors influencing prices.
# - However, 14% of the variance remains unexplained, possibly due to missing features (e.g., economic trends, interest rates) or randomness in the data.

# In[ ]:





# ### 4.2 Residual Analysis
# Analyze residuals to assess the model's performance.

# In[22]:


# Plot residuals
residuals = y_test - y_pred
plt.figure(figsize=(18, 6))
sns.histplot(residuals, kde=True, bins=30)
plt.title("Residual Distribution")
plt.xlabel("Residuals")
plt.ylabel("Frequency")
plt.show()


# ***Insights***:
# 
# The residuals (difference between actual and predicted prices) form a normal distribution:
# This indicates that the model's errors are evenly distributed, without significant bias (e.g., overestimating or underestimating prices consistently).

# In[ ]:





# # 5. Business Insights

# ### 5.1 Feature Importance
# For linear regression, feature importance can be interpreted using coefficients.

# In[23]:


# Create a DataFrame of feature importance
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': lr.coef_
}).sort_values(by='Coefficient', ascending=False)


# In[24]:


feature_importance


# ### 5.2 Summary of overal Insights
# - Larger properties (SizeInSqFt) have the highest positive impact on house prices.
# - The number of bedrooms(Bedrooms) and bathrooms(Bathrooms) significantly increase prices.
# - High-rated schools (NearbySchools) are a key driver of pricing in family-friendly neighborhoods.

# In[ ]:





# In[ ]:




