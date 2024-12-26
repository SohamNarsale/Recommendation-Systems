import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report


import os
os.environ["LOKY_MAX_CPU_COUNT"] = "4"  # Limit to 1 core (adjust as needed)


# Load dataset
data = load_breast_cancer()
X = data.data
y = data.target

# Split dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train a RandomForest model for predictions (as in your original setup)
model = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='logloss',
    use_label_encoder=False,
    random_state=42
)
model.fit(X_train, y_train)

# Define recommendations (you can extend this list as per your use case)
recommendations = {
    0: 'Routine Monitoring',
    1: 'Lifestyle Changes',
    2: 'Follow-Up Tests',
    3: 'Minimally Invasive Procedure',
    4: 'Surgery',
    5: 'Chemotherapy',
    6: 'Targeted Therapy',
    7: 'Hormone Therapy',
    8: 'Clinical Trials'
}

# Train a K-Means Clustering Model for Recommendations
num_clusters = len(recommendations)  # You can modify this based on the number of recommendation categories
kmeans = KMeans(n_clusters=num_clusters, n_init=10, random_state=42)
kmeans.fit(X_train)

# Apply PCA to reduce the data to 2D for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_train)

# Get cluster labels for each data point
cluster_labels = kmeans.predict(X_train)

# Plot the clusters
plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, cmap='viridis', marker='o', s=50)
plt.title('KMeans Clusters Visualized with PCA')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')

# Add the centroids of the clusters to the plot
centroids = pca.transform(kmeans.cluster_centers_)
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X', s=100, label='Centroids')

# Show the plot
plt.legend()
plt.colorbar(label='Cluster Label')
plt.show()

# Predict the recommendation cluster for a test case
test_index = 1  # Change this index to test other samples
test_sample = X_test[test_index].reshape(1, -1)
predicted_class = model.predict(test_sample)[0]
actual_class = y_test[test_index]
cluster_label = kmeans.predict(test_sample)[0]
recommendation = recommendations[cluster_label]

# Output Test Case Details
test_case_original = scaler.inverse_transform(test_sample)[0]
test_case_df = pd.DataFrame({
    'Feature': data.feature_names,
    'Value': test_case_original
})

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy*100:.2f}%")
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=data.target_names))

print("###########################################################################################################")
print("\nTest Case Details:")
print(test_case_df)

print("\nPrediction Results:")
print(f"Predicted class: {data.target_names[predicted_class]}")
print(f"Actual class: {data.target_names[actual_class]}")
print(f"Recommendation: {recommendation}")
