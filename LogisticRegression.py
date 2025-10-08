import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

df = pd.read_csv("datasets/wildfires_training.csv")
test_df = pd.read_csv("datasets/wildfires_test.csv")
data = df.to_numpy()
test_data = test_df.to_numpy()

X = data[:, 1:]
y = data[:, 0]

X_test = test_data[:, 1:]
Y_test = test_data[:, 0]

model = LogisticRegression()
model.fit(X, y)

predicted_Y = model.predict(X_test)

accuracy = accuracy_score(Y_test, predicted_Y)
conf_matrix = confusion_matrix(Y_test, predicted_Y)

print("Accuracy:", accuracy)
print("Confusion Matrix:\n", conf_matrix)