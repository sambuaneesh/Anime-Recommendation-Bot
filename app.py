from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from fuzzywuzzy import fuzz

app = Flask(__name__)

anime_df = pd.read_csv("data/anime.csv")
anime_df = anime_df.dropna()
anime_df = anime_df.drop_duplicates()

rating_df = pd.read_csv("data/rating.csv")
rating_df = rating_df.dropna()
rating_df = rating_df.drop_duplicates()

anime_ratings_df = pd.merge(rating_df, anime_df, on="anime_id")

tfidf = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf.fit_transform(anime_df["genre"])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

@app.route('/recommend', methods=['GET'])
def recommend():
    anime_name = request.args.get('anime_name')
    if not anime_name:
        return jsonify({"error": "Anime name not provided"})
    n = request.args.get('n', default=5, type=int)
    
    anime_titles = anime_df["name"].values
    matching_scores = [(idx, fuzz.ratio(anime_name.lower(), anime_title.lower())) for idx, anime_title in enumerate(anime_titles)]
    matching_scores = sorted(matching_scores, key=lambda x: x[1], reverse=True)
    closest_match_idx = matching_scores[0][0]
    closest_match_score = matching_scores[0][1]
    if closest_match_score < 60:
        return jsonify({"error": "Anime not found. Please try a different name."})
    
    sim_scores = list(enumerate(cosine_sim[closest_match_idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    anime_indices = [i[0] for i in sim_scores[1:n+1]]
    
    recommended_anime_ratings = anime_ratings_df.loc[anime_ratings_df["name"].isin(anime_df["name"].iloc[anime_indices])]
    recommended_anime_ratings = recommended_anime_ratings.groupby("name").agg({"rating_x": "mean"})
    recommended_anime_ratings_dict = recommended_anime_ratings.sort_values(by="rating_x", ascending=False).head(n).to_dict()
    
    return jsonify(recommended_anime_ratings_dict)

if __name__ == '__main__':
    app.run(debug=True)
