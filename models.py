import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
import lightgbm as lgb
import joblib

# Load dataset
csv_file_path = 'E:\\finalGPbegad\\blob_analysis_results.csv'
df = pd.read_csv(csv_file_path)
df.info()

# Function to calculate the percentage of outliers
def outlier_percent(data):
    numeric_columns = data.select_dtypes(include=[np.number])
    Q1 = numeric_columns.quantile(0.25)
    Q3 = numeric_columns.quantile(0.75)
    IQR = Q3 - Q1
    minimum = Q1 - (1.5 * IQR)
    maximum = Q3 + (1.5 * IQR)
    num_outliers = ((numeric_columns < minimum) | (numeric_columns > maximum)).sum().sum()
    num_total = numeric_columns.count().sum()
    return (num_outliers / num_total) * 100

print(f"Outlier Percentage: {outlier_percent(df):.2f}%")

# Clean the data: replace infinite values with NaN, then drop rows with NaN or infinite values
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)

# Encoding Prediction column
df["Prediction"].replace(to_replace="Debris", value=0, inplace=True)
df["Prediction"].replace(to_replace="Star", value=1, inplace=True)

# Verify there are no infinite values
assert np.isfinite(df[['Mean Intensity', 'Std Deviation', 'Area', 'LBP Mean', 'LBP Std', 'Imain1', 'Imain2', 'Final Inertia']]).all().all()

# Scale the selected columns
scaler = StandardScaler()
df[['Mean Intensity', 'Std Deviation', 'Area', 'LBP Mean', 'LBP Std', 'Imain1', 'Imain2', 'Final Inertia']] = scaler.fit_transform(
    df[['Mean Intensity', 'Std Deviation', 'Area', 'LBP Mean', 'LBP Std', 'Imain1', 'Imain2', 'Final Inertia']])

# Calculate and plot correlation matrix
correlation_matrix = df.drop(["Blob ID"], axis=1).corr()
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))

plt.figure(figsize=(14, 14))
sns.heatmap(data=correlation_matrix, annot=True, cmap='coolwarm', center=0, mask=mask)
plt.title('Correlation Matrix')
plt.show()

# Shuffle the dataset
df = df.sample(frac=1, random_state=42)

# Split data into features and target
X = df.drop(['Blob ID', 'Prediction'], axis=1)
y = df['Prediction']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model_names = []
accuracies = []

# Support Vector Classifier (SVC)
svc_model = SVC()
svc_model.fit(X_train, y_train)
svc_predictions = svc_model.predict(X_test)
svc_accuracy = accuracy_score(y_test, svc_predictions)
print(f"SVC Accuracy: {svc_accuracy:.4f}")
model_names.append("SVC")
accuracies.append(svc_accuracy)

# k-Nearest Neighbors (k-NN)
knn_model = KNeighborsClassifier()
knn_model.fit(X_train, y_train)
knn_accuracy = knn_model.score(X_test, y_test)
print(f"KNN Accuracy: {knn_accuracy:.4f}")
model_names.append("KNN")
accuracies.append(knn_accuracy)

# Logistic Regression
lr_model = LogisticRegression()
lr_model.fit(X_train, y_train)
lr_accuracy = lr_model.score(X_test, y_test)
print(f"Logistic Regression Accuracy: {lr_accuracy:.4f}")
model_names.append("LR")
accuracies.append(lr_accuracy)

# # Random Forest Classifier
# rf_model = RandomForestClassifier()
# rf_model.fit(X_train, y_train)
# rf_accuracy = rf_model.score(X_test, y_test)
# print(f"Random Forest Accuracy: {rf_accuracy:.4f}")
# model_names.append("RF")
# accuracies.append(rf_accuracy)

# # Decision Tree Classifier
# dt_model = DecisionTreeClassifier()
# dt_model.fit(X_train, y_train)
# dt_accuracy = dt_model.score(X_test, y_test)
# print(f"Decision Tree Accuracy: {dt_accuracy:.4f}")
# model_names.append("DT")
# accuracies.append(dt_accuracy)

# XGBoost Classifier
xgb_model = xgb.XGBClassifier()
xgb_model.fit(X_train, y_train)
xgb_accuracy = xgb_model.score(X_test, y_test)
print(f"XGBoost Accuracy: {xgb_accuracy:.4f}")
model_names.append("XGB")
accuracies.append(xgb_accuracy)

# LightGBM Classifier
lgb_model = lgb.LGBMClassifier(verbosity=-1)
lgb_model.fit(X_train, y_train)
lgb_accuracy = lgb_model.score(X_test, y_test)
print(f"LightGBM Accuracy: {lgb_accuracy:.4f}")
model_names.append("LGBM")
accuracies.append(lgb_accuracy)

# Plot model accuracies
plt.figure(figsize=(10, 8))
plt.bar(model_names, accuracies, color='skyblue')
plt.title('Model Comparison')
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.show()

plt.figure(figsize=(10, 10))
plt.plot(model_names, accuracies, 'r*-')
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.title('Model Accuracy Comparison')
plt.show()

# Save and load the best model (Random Forest in this case)
joblib.dump(rf_model, 'best_model.pkl')
loaded_model = joblib.load('best_model.pkl')

# Make predictions with the loaded model
predictions = loaded_model.predict(df.drop(['Blob ID', 'Prediction', 'Final Inertia'], axis=1))
print(predictions)
