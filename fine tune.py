import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
import joblib

data = pd.read_csv('Dataset/modified_iris.csv')
X = data.drop(['variety'], axis=1)
y = data['variety']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

param_grid = {
    'C': [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100, 300],
    'kernel': ['linear', 'rbf', 'poly'],
    'degree': [1, 2, 3, 4, 5],
}

svm_model = SVC(random_state=42)

grid_search = GridSearchCV(svm_model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_

best_model = SVC(**best_params, random_state=42)
best_model.fit(X_train, y_train)

y_pred = best_model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
print(f"Best hyperparameters: {best_params}")
print(f"Accuracy: {acc}")
print(f"Confusion matrix:\n{cm}")

joblib.dump(best_model, 'Model/best_iris_classification_model.pkl')
