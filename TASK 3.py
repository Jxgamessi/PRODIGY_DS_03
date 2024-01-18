import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn import tree
import matplotlib.pyplot as plt

file_path = "C:/Users/Lenovo/Desktop/Prodigy Infotech/bank_data.csv"
data = pd.read_csv(file_path)

print(data.head())


features = ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan', 'contact', 'day', 'month', 'duration', 'campaign', 'pdays', 'previous', 'poutcome']
X = data[features]
y = data['y']

X = pd.get_dummies(X)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


clf = DecisionTreeClassifier()


clf.fit(X_train, y_train)


y_pred = clf.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print(f"Confusion Matrix:\n{conf_matrix}")


plt.figure(figsize=(20, 10))
tree.plot_tree(clf, filled=True, feature_names=X.columns, class_names=['No Purchase', 'Purchase'])
plt.show()


new_data = pd.DataFrame({
    'age': [30],
    'job': ['management'],
    'marital': ['single'],
    'education': ['tertiary'],
    'default': ['no'],
    'balance': [5000],
    'housing': ['no'],
    'loan': ['no'],
    'contact': ['cellular'],
    'day': [15],
    'month': ['may'],
    'duration': [200],
    'campaign': [2],
    'pdays': [10],
    'previous': [3],
    'poutcome': ['success']
})


new_data = pd.get_dummies(new_data)


missing_cols = set(X.columns) - set(new_data.columns)
for col in missing_cols:
    new_data[col] = 0


new_data = new_data[X.columns]


prediction = clf.predict(new_data)


if prediction[0] == 1:
    print("The customer is likely to purchase.")
else:
    print("The customer is unlikely to purchase.")
