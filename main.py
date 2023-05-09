# Import necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from fuzzywuzzy import fuzz

# Load the anime dataset
anime_df = pd.read_csv("data/anime.csv")

# Check for missing values and handle them
anime_df = anime_df.dropna()

# Check for duplicate values and remove them if any
anime_df = anime_df.drop_duplicates()

# Load the rating dataset
rating_df = pd.read_csv("data/rating.csv")

# Check for missing values and handle them
rating_df = rating_df.dropna()

# Check for duplicate values and remove them if any
rating_df = rating_df.drop_duplicates()

# Merge the ratings dataframe with the anime dataframe to get the names of the anime
anime_ratings_df = pd.merge(rating_df, anime_df, on="anime_id")

# Create a TfidfVectorizer object
tfidf = TfidfVectorizer(stop_words="english")

# Fit and transform the anime genres into a matrix of TF-IDF features
tfidf_matrix = tfidf.fit_transform(anime_df["genre"])

# Calculate the cosine similarity matrix of the TF-IDF features
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

def get_recommendations(title, n=5):
    # Use fuzzy string matching to find the closest matching anime title
    anime_titles = anime_df["name"].values
    matching_scores = [(idx, fuzz.ratio(title.lower(), anime_title.lower())) for idx, anime_title in enumerate(anime_titles)]
    matching_scores = sorted(matching_scores, key=lambda x: x[1], reverse=True)
    closest_match_idx = matching_scores[0][0]
    closest_match_score = matching_scores[0][1]
    print("Expected Input:",closest_match_idx,anime_df["name"].iloc[closest_match_idx])
    # Check if the similarity score of the closest match is below a certain threshold
    if closest_match_score < 60:
        return "Anime not found. Please try a different name."
    
    # Get the cosine similarity scores of all anime compared to the given anime
    sim_scores = list(enumerate(cosine_sim[closest_match_idx]))

    # Sort the cosine similarity scores in descending order
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the indices of the top n most similar anime
    anime_indices = [i[0] for i in sim_scores[1:n+1]]
    
    # Get the ratings for each recommended anime
    recommended_anime_ratings = anime_ratings_df.loc[anime_ratings_df["name"].isin(anime_df["name"].iloc[anime_indices])]

    # Calculate the average rating for each recommended anime
    recommended_anime_ratings = recommended_anime_ratings.groupby("name").agg({"rating_x": "mean"})
    
    # Plot the distribution of ratings for each recommended anime
    for anime_name in recommended_anime_ratings.index:
        anime_ratings = anime_ratings_df.loc[anime_ratings_df["name"] == anime_name]["rating_x"]
        plt.hist(anime_ratings, alpha=0.5, label=anime_name)
    plt.title("Distribution of ratings for recommended anime")
    plt.xlabel("Rating")
    plt.ylabel("Frequency")
    plt.legend(loc="upper right")
    plt.show()

    return recommended_anime_ratings.sort_values(by="rating_x", ascending=False).head(n)

print(get_recommendations("steins gate"))
