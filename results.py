import joblib
import numpy as np
import pandas as pd
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
import json
from sklearn.model_selection import train_test_split

class_names = ["Setosa", "Versicolor", "Virginica"]
# Load the SVM model
svm_model = joblib.load('Model/iris classification best hyperparameters.pkl')

# Load the test dataset
test_data = pd.read_csv('Dataset/modified_iris.csv')

# Extract the features and labels from the test dataset
X = test_data.drop('variety', axis=1)
y = test_data['variety']

# Extract the training features
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Predict the labels for the test dataset
y_pred = svm_model.predict(X_test)

print(y_pred)
# Calculate accuracy
accuracy = metrics.accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)

# Calculate F1 score, precision, recall for each class
class_metrics = metrics.classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
print('Classification Report:\n', class_metrics)

# Calculate confusion matrix
confusion_matrix = metrics.confusion_matrix(y_test, y_pred)
print('Confusion Matrix:\n', confusion_matrix)

# Create a DataFrame from the confusion matrix
cm_df = pd.DataFrame(confusion_matrix, index=class_names, columns=class_names)

# Save the confusion matrix as an image
plt.figure(figsize=(12, 10))
sns.heatmap(cm_df, annot=True, fmt='d', cmap='Greens')
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.savefig('Results/confusion_matrix_Iris.png')
plt.show()

# Save the classification report as a JSON file
with open('Results/Classification_report_Iris.json', 'w') as json_file:
    json.dump(class_metrics, json_file, indent=4)
