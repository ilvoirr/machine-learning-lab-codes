from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
# Implement Naïve Bayesian classification
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)
y_pred = nb_model.predict(X_test)
# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)