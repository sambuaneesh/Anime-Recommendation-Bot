from flask import Flask, render_template, request, jsonify, send_file
import matplotlib
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from fuzzywuzzy import fuzz
import io
import base64
import matplotlib.pyplot as plt
matplotlib.use('Agg')
from flask_cors import CORS

app = Flask(__name__)
CORS(app)


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

@app.route("/")
def serve_html():
    return send_file("index.html")

# Generate the histogram plot
# Generate the histogram plot
def generate_histogram(anime_name_ratings_dict):
    plt.clf()  # Clear the current figure
    
    for anime_name in anime_name_ratings_dict.index:
        anime_ratings = anime_ratings_df.loc[anime_ratings_df["name"] == anime_name]["rating_x"]
        plt.hist(anime_ratings, alpha=0.5, label=anime_name)
    plt.title("Distribution of ratings for recommended anime")
    plt.xlabel("Rating")
    plt.ylabel("Frequency")
    plt.legend(loc="upper right")
    
    # Save the plot as a PNG file
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    
    # Encode the PNG file as a base64 string
    image_string = base64.b64encode(buffer.read()).decode()
    
    # Return the base64 string
    return image_string


@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.json
    anime_name = data.get('anime_name')
    if not anime_name:
        return jsonify({"error": "Anime name not provided"})
    n = data.get('n', 5)
    n = int(n)
    # Find the closest match based on fuzzy string matching
    anime_titles = anime_df["name"].values
    matching_scores = [(idx, fuzz.ratio(anime_name.lower(), anime_title.lower())) for idx, anime_title in enumerate(anime_titles)]
    matching_scores = sorted(matching_scores, key=lambda x: x[1], reverse=True)
    closest_match_idx = matching_scores[0][0]
    closest_match_score = matching_scores[0][1]
    if closest_match_score < 60:
        return jsonify({"error": "Anime not found. Please try a different name."})
    
    # Calculate cosine similarity and recommend anime based on genre
    sim_scores = list(enumerate(cosine_sim[closest_match_idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    anime_indices = [i[0] for i in sim_scores[1:int(n)+1]]
    recommended_anime_ratings = anime_ratings_df.loc[anime_ratings_df["name"].isin(anime_df["name"].iloc[anime_indices])]
    recommended_anime_ratings = recommended_anime_ratings.groupby("name").agg({"rating_x": "mean"})
    recommended_anime_ratings_dict = recommended_anime_ratings.sort_values(by="rating_x", ascending=False).head(n).to_dict()
    
    # Generate the histogram plot and return the base64-encoded image string
    image_string = generate_histogram(recommended_anime_ratings)
    return jsonify({'recommended_anime_ratings_dict': recommended_anime_ratings_dict, 'image_string': image_string})

if __name__ == '__main__':
    app.run(debug=True)
