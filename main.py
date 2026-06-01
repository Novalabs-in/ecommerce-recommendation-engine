import numpy as np
from scipy.sparse.linalg import svds

class ProductRecommender:
    """
    Ecommerce Product Recommendation Engine
    Utilizes Singular Value Decomposition (SVD) matrix factorization for user recommendation matches.
    """
    def __init__(self):
        pass

    def factorize_ratings(self, ratings_matrix, k=2):
        # Convert ratings to float sparse-like array
        ratings = ratings_matrix.astype(float)
        user_ratings_mean = np.mean(ratings, axis=1)
        ratings_demeaned = ratings - user_ratings_mean.reshape(-1, 1)
        
        # Factor SVD
        U, sigma, Vt = svds(ratings_demeaned, k=k)
        sigma = np.diag(sigma)
        
        predicted_ratings = np.dot(np.dot(U, sigma), Vt) + user_ratings_mean.reshape(-1, 1)
        return predicted_ratings

if __name__ == "__main__":
    # 4 users, 5 products
    ratings = np.array([
        [5, 3, 0, 1, 4],
        [4, 0, 0, 1, 4],
        [1, 1, 0, 5, 5],
        [1, 0, 0, 4, 0]
    ])
    recommender = ProductRecommender()
    preds = recommender.factorize_ratings(ratings, k=2)
    print("Predicted Product Ratings Matrix:")
    print(np.round(preds, 2))
