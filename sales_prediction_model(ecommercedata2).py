
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load the dataset
file_path = 'path/to/ecommerce_product_dataset.csv'
data = pd.read_csv(file_path)

# Identify the features and target variable
target_column = 'Price'
feature_columns = data.columns.drop(target_column)

# Separate features and target variable
X = data[feature_columns]
y = data[target_column]

# Fill missing values if necessary
for column in X.select_dtypes(include=['float64', 'int64']).columns:
    X.loc[:, column] = X[column].fillna(X[column].mean())

for column in X.select_dtypes(include=['object']).columns:
    X.loc[:, column] = X[column].fillna(X[column].mode()[0])

# Encode categorical variables
categorical_features = X.select_dtypes(include=['object']).columns.tolist()
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', encoder, categorical_features)
    ], remainder='passthrough'
)

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
future_predictions = rf_model.predict(X_test)
print("\nFuture predictions:")
print(future_predictions[:10])
