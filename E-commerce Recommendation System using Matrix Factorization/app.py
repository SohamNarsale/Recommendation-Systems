from surprise import SVD, Dataset, Reader
from surprise.model_selection import train_test_split
from surprise.accuracy import rmse
import pandas as pd
import numpy as np

# Create a sample user-item rating matrix as a DataFrame
df = pd.read_csv(r"csv_path")

# Define a Reader and load the data into Surprise's format
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(df[['User ID', 'Clothing ID', 'Rating']], reader)

# Split the dataset into training and testing sets
trainset, testset = train_test_split(data, test_size=0.25, random_state=42)

# Use the SVD algorithm for matrix factorization
model = SVD()

# Train the model on the training set
model.fit(trainset)

# Test the model on the test set
predictions = model.test(testset)

# Calculate and print the RMSE for evaluation
print("Root Mean Squared Error (RMSE):")
rmse(predictions)

# Full recommendations for a specific user
def get_recommendations_for_user(model, user_id, num_recommendations=5):
    # Get a list of all item IDs
    all_items = df['Clothing ID'].unique()
    # Predict ratings for all items the user hasn't rated yet
    rated_items = df[df['User ID'] == user_id]['Clothing ID'].values
    recommendations = [
        (item_id, model.predict(user_id, item_id).est)
        for item_id in all_items if item_id not in rated_items
    ]
    # Sort by predicted rating in descending order
    recommendations.sort(key=lambda x: x[1], reverse=True)
    return recommendations[:num_recommendations]

# Example: Get top 5 recommendations for user 1
user_recommendations = get_recommendations_for_user(model, user_id=1, num_recommendations=5)
print("\nTop 5 recommendations for user 1:")
for item_id, predicted_rating in user_recommendations:
    print(f"Item {item_id}: Predicted rating {predicted_rating:.2f}")

Displaying the original user-item matrix (R)
print("\nOriginal User-Item Rating Matrix (R):")
R_matrix = df.pivot(index='User ID', columns='Clothing ID', values='Rating').fillna(0).values
print(R_matrix)

Display the factorized matrices (P and Q)
print("\nUser Factor Matrix:")
P_matrix = model.pu
print(P_matrix)

print("\nItem Factor Matrix:")
Q_matrix = model.qi
print(Q_matrix)

# Reconstruct the full matrix from pu and qi
reconstructed_matrix = np.dot(P_matrix, Q_matrix.T)
print("\nPredicted User-Item Matrix:")
print(np.round(reconstructed_matrix, 2))
