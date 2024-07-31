import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load dataset
file_path = 'path/to/BigMart Sales Data.csv'
data = pd.read_csv(file_path)

# Display the first few rows of the dataset
print("First few rows of the dataset:")
print(data.head())

# Check for missing values
print("\nMissing values in the dataset:")
print(data.isnull().sum())

# Handling missing values
# Fill missing Item_Weight with mean value
data['Item_Weight'] = data['Item_Weight'].fillna(data['Item_Weight'].mean())
# Fill missing Outlet_Size with mode value
data['Outlet_Size'] = data['Outlet_Size'].fillna(data['Outlet_Size'].mode()[0])

# Encode categorical variables
categorical_features = ['Item_Fat_Content', 'Item_Type', 'Outlet_Identifier', 
                        'Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type']
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', encoder, categorical_features)
    ], remainder='passthrough'
)

# Separate features and target variable
X = data.drop(['Item_Outlet_Sales', 'Item_Identifier'], axis=1)
y = data['Item_Outlet_Sales']

# Preprocess features
X = preprocessor.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Linear Regression model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_predictions = lr_model.predict(X_test)

# Train Random Forest Regressor model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_test)

# Evaluate model performance
def evaluate_model(predictions, true_values):
    mae = mean_absolute_error(true_values, predictions)
    mse = mean_squared_error(true_values, predictions)
    r2 = r2_score(true_values, predictions)
    return mae, mse, r2

lr_mae, lr_mse, lr_r2 = evaluate_model(lr_predictions, y_test)
rf_mae, rf_mse, rf_r2 = evaluate_model(rf_predictions, y_test)

print(f"Linear Regression - MAE: {lr_mae}, MSE: {lr_mse}, R2: {lr_r2}")
print(f"Random Forest Regression - MAE: {rf_mae}, MSE: {rf_mse}, R2: {rf_r2}")

# Making predictions
# For demonstration, we'll use the test set for future predictions
future_predictions = rf_model.predict(X_test)
print("\nFuture predictions:")
print(future_predictions[:10])
