from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import mean_squared_error
# Load dataset
housing = fetch_california_housing()
X = housing.data
y = housing.target
# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
random_state=42)
# Train linear regression model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
y_pred = lr_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("MSE without scaling:", mse)
# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
# Train linear regression model with scaled features
lr_model.fit(X_train_scaled, y_train)
y_pred_scaled = lr_model.predict(X_test_scaled)
mse_scaled = mean_squared_error(y_test, y_pred_scaled)
print("MSE with scaling:", mse_scaled)
# Impact of outliers
# Add artificial outliers
X_train_outliers = np.copy(X_train)
y_train_outliers = np.copy(y_train)
X_train_outliers[:5] = X_train_outliers[:5] + 100 # Adding large values
y_train_outliers[:5] = y_train_outliers[:5] + 50
# Train model with outliers
lr_model.fit(X_train_outliers, y_train_outliers)
y_pred_outliers = lr_model.predict(X_test)
mse_outliers = mean_squared_error(y_test, y_pred_outliers)
print("MSE with outliers:", mse_outliers)