import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def hybrid_recommendation_system(movie_name, movies_data, n_neighbors=3):
    """
    Recommend movies similar to the given movie using both collaborative and content-based filtering.

    Parameters:
    - movie_name (str): The name of the movie to find recommendations for.
    - movies_data (pandas dataframe): Dataframe containing movie details.
    - n_neighbors (int): The number of similar movies to recommend.

    Returns:
    - list of str: List of recommended movie names.
    """
    
    # Convert the movies data to DataFrame
    movie_ratings = movies_data[['name', 'rating']]
    
    # Convert movie ratings to numpy array for collaborative filtering
    ratings_matrix = movie_ratings['rating'].values.reshape(-1, 1)

    # Fit the K-Nearest Neighbors model for collaborative filtering using cosine similarity
    knn_collab = NearestNeighbors(n_neighbors=n_neighbors, metric='cosine')
    knn_collab.fit(ratings_matrix)

    # Find the index of the given movie in ratings DataFrame
    movie_index_collab = movie_ratings[movie_ratings['name'] == movie_name].index[0]
    
    # Get nearest neighbors for collaborative filtering
    _, indices_collab = knn_collab.kneighbors(ratings_matrix[movie_index_collab].reshape(1, -1))

    # Content-based Filtering: Extract features for TF-IDF Vectorization
    features_df = pd.DataFrame(movies_data['name'])
    features_df['features'] = movies_data[['genre','directors','casts']].agg(", ".join, axis=1)


    # Vectorize the movie features using TF-IDF
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform(features_df['features'])

    # Compute cosine similarities between movies based on their features
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    
    # Find the index of the given movie in features DataFrame
    movie_index_content = features_df[features_df['name'] == movie_name].index[0]
    
    # Get most similar movies based on content-based filtering
    similar_indices_content = cosine_sim[movie_index_content].argsort()[-n_neighbors-1:-1][::-1]

    # Combine collaborative and content-based recommendations
    recommended_movies = set()
    
    # Add collaborative-based recommendations
    for idx in indices_collab[0]:
        recommended_movies.add(movie_ratings.iloc[idx]['name'])
    
    # Add content-based recommendations
    for idx in similar_indices_content:
        recommended_movies.add(features_df.iloc[idx]['name'])

    # Remove the movie itself from recommendations (if exists)
    recommended_movies.discard(movie_name)
    
    return list(recommended_movies)

if __name__ == "__main__":
    print("Hybrid Recommendation System\n\nMovies")
    movies_data = pd.read_csv(r"datasets\RS Practical 3\IMDB Top 250 Movies.csv")
    print(f"{movies_data["name"].sample(10)}")
    user_movie = input("\n\nEnter movie: ")
    recommended_movies = hybrid_recommendation_system(user_movie, movies_data)
    print(f"\nMovies similar to {user_movie}:")
    for movie in recommended_movies:
        print(f"- {movie}")
