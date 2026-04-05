# Import libraries
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
# Load dataset
data = load_breast_cancer()
X = data.data
y = data.target
# Split dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(
X, y, test_size=0.2, random_state=42
)
# -----------------------------
# Linear Kernel SVM
# -----------------------------
svm_linear = SVC(kernel=&#39;linear&#39;, C=1)
svm_linear.fit(X_train, y_train)
# Prediction
y_pred_linear = svm_linear.predict(X_test)
# Evaluation
print(&quot;Linear SVM Accuracy:&quot;, accuracy_score(y_test, y_pred_linear))
print(&quot;Confusion Matrix (Linear):&quot;)
print(confusion_matrix(y_test, y_pred_linear))

# -----------------------------
# RBF Kernel SVM
# -----------------------------
svm_rbf = SVC(kernel=&#39;rbf&#39;, C=1, gamma=0.1)
svm_rbf.fit(X_train, y_train)
# Prediction
y_pred_rbf = svm_rbf.predict(X_test)
# Evaluation
print(&quot;\nRBF SVM Accuracy:&quot;, accuracy_score(y_test, y_pred_rbf))
print(&quot;Confusion Matrix (RBF):&quot;)
print(confusion_matrix(y_test, y_pred_rbf))

# -----------------------------
# Hyperparameter Tuning
# -----------------------------
param_grid = {
&#39;C&#39;: [0.1, 1, 10],
&#39;gamma&#39;: [0.01, 0.1, 1]
}
grid = GridSearchCV(SVC(kernel=&#39;rbf&#39;), param_grid, cv=5)
grid.fit(X_train, y_train)
print(&quot;\nBest Parameters:&quot;, grid.best_params_)
