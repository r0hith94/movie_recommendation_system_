"""
Movie Recommendation System - Complete Flask Web Application
With TMDb API Integration for Movie Posters
"""

from flask import Flask, render_template, request, jsonify
import pandas as pd
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import logging
import requests
from functools import lru_cache

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# TMDb API Configuration
# REPLACE THIS WITH YOUR ACTUAL API KEY FROM TMDb
TMDB_API_KEY = "ab678aa4ce9dc609001b425f8d262d78"
TMDB_BASE_URL = "https://api.themoviedb.org/3"
TMDB_IMAGE_BASE_URL = "https://image.tmdb.org/t/p/w500"

# Global variables
movies_data = None
similarity_matrix = None
movie_titles = None


@lru_cache(maxsize=1000)
def get_movie_poster(movie_title):
    """
    Fetch movie poster from TMDb API with retry logic and rate limiting
    
    Args:
        movie_title (str): Movie title to search
        
    Returns:
        str: Poster URL or placeholder image URL
    """
    import time
    
    max_retries = 3
    retry_delay = 0.5  # seconds between retries
    
    for attempt in range(max_retries):
        try:
            # Clean the movie title
            clean_title = movie_title.strip()
            
            # Try exact search first
            search_url = f"{TMDB_BASE_URL}/search/movie"
            params = {
                'api_key': TMDB_API_KEY,
                'query': clean_title,
                'page': 1
            }
            
            # Make request with longer timeout and session
            response = requests.get(
                search_url, 
                params=params, 
                timeout=10,
                headers={'Connection': 'close'}  # Prevent connection reuse issues
            )
            
            data = response.json()
            
            # Check if results found
            if data.get('results') and len(data['results']) > 0:
                poster_path = data['results'][0].get('poster_path')
                
                if poster_path:
                    return f"{TMDB_IMAGE_BASE_URL}{poster_path}"
            
            # If no results, try removing year from title
            if '(' in clean_title:
                title_without_year = clean_title.split('(')[0].strip()
                params['query'] = title_without_year
                
                response = requests.get(
                    search_url, 
                    params=params, 
                    timeout=10,
                    headers={'Connection': 'close'}
                )
                data = response.json()
                
                if data.get('results') and len(data['results']) > 0:
                    poster_path = data['results'][0].get('poster_path')
                    if poster_path:
                        return f"{TMDB_IMAGE_BASE_URL}{poster_path}"
            
            # If we get here, no poster found but request succeeded
            logger.warning(f"No poster found for: {movie_title}")
            return "https://via.placeholder.com/500x750/667eea/ffffff?text=No+Poster"
            
        except requests.exceptions.ConnectionError as e:
            # Connection error - retry with delay
            if attempt < max_retries - 1:
                logger.warning(f"Connection error for {movie_title}, retrying in {retry_delay}s... (attempt {attempt + 1}/{max_retries})")
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
                continue
            else:
                logger.error(f"Failed to fetch poster for {movie_title} after {max_retries} attempts")
                return "https://via.placeholder.com/500x750/667eea/ffffff?text=No+Poster"
                
        except Exception as e:
            logger.error(f"Error fetching poster for {movie_title}: {str(e)}")
            return "https://via.placeholder.com/500x750/667eea/ffffff?text=No+Poster"
    
    # Fallback
    return "https://via.placeholder.com/500x750/667eea/ffffff?text=No+Poster"


def load_data():
    """Load and preprocess movie data at startup"""
    global movies_data, similarity_matrix, movie_titles
    
    try:
        logger.info("Loading movie dataset...")
        
        # Load the CSV file - UPDATE THIS PATH
        movies_data = pd.read_csv('movies.csv')
        logger.info(f"Loaded {len(movies_data)} movies")
        
        # Selected features for recommendation
        selected_features = ['genres', 'keywords', 'tagline', 'cast', 'director']
        
        # Fill missing values
        for feature in selected_features:
            movies_data[feature] = movies_data[feature].fillna('')
        
        # Combine all features
        combined_features = (
            movies_data['genres'] + ' ' +
            movies_data['keywords'] + ' ' +
            movies_data['tagline'] + ' ' +
            movies_data['cast'] + ' ' +
            movies_data['director']
        )
        
        # Create TF-IDF vectors
        logger.info("Creating TF-IDF vectors...")
        vectorizer = TfidfVectorizer()
        feature_vectors = vectorizer.fit_transform(combined_features)
        
        # Compute similarity matrix
        logger.info("Computing similarity matrix...")
        similarity_matrix = cosine_similarity(feature_vectors)
        
        # Store movie titles for fuzzy matching
        movie_titles = movies_data['title'].tolist()
        
        logger.info("Data loaded and processed successfully!")
        
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise


def get_recommendations(movie_name, top_n=30):
    """Get movie recommendations with poster URLs"""
    import time
    
    try:
        # Find close match using fuzzy matching
        close_matches = difflib.get_close_matches(movie_name, movie_titles, n=1, cutoff=0.6)
        
        if not close_matches:
            return {
                'status': 'error',
                'message': f'Movie "{movie_name}" not found. Please check the spelling.',
                'matched_movie': None,
                'matched_poster': None,
                'recommendations': []
            }
        
        # Get the matched movie title
        matched_title = close_matches[0]
        
        # Get poster for matched movie
        matched_poster = get_movie_poster(matched_title)
        
        # Find the index of the movie
        movie_index = movies_data[movies_data['title'] == matched_title]['index'].values[0]
        
        # Get similarity scores
        similarity_scores = list(enumerate(similarity_matrix[movie_index]))
        
        # Sort by similarity score
        sorted_movies = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
        
        # Get top N recommendations (skip first as it's the input movie itself)
        recommendations = []
        for i, (idx, score) in enumerate(sorted_movies[1:top_n+1]):
            movie_title = movies_data[movies_data['index'] == idx]['title'].values[0]
            
            # Get poster URL for this movie
            poster_url = get_movie_poster(movie_title)
            
            recommendations.append({
                'rank': i + 1,
                'title': movie_title,
                'similarity_score': round(float(score), 4),
                'poster_url': poster_url
            })
            
            # Add small delay every 5 movies to avoid rate limiting
            if (i + 1) % 5 == 0:
                time.sleep(0.2)
        
        return {
            'status': 'success',
            'message': 'Recommendations generated successfully',
            'matched_movie': matched_title,
            'matched_poster': matched_poster,
            'recommendations': recommendations
        }
        
    except Exception as e:
        logger.error(f"Error generating recommendations: {str(e)}")
        return {
            'status': 'error',
            'message': f'Error: {str(e)}',
            'matched_movie': None,
            'matched_poster': None,
            'recommendations': []
        }


# Routes
@app.route('/')
def home():
    """Render the home page"""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """API endpoint to get recommendations"""
    try:
        data = request.get_json()
        movie_name = data.get('movie_name', '').strip()
        top_n = data.get('top_n', 20)
        
        if not movie_name:
            return jsonify({
                'status': 'error',
                'message': 'Please enter a movie name'
            }), 400
        
        # Get recommendations
        result = get_recommendations(movie_name, top_n)
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in predict endpoint: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f'Server error: {str(e)}'
        }), 500


@app.route('/autocomplete', methods=['GET'])
def autocomplete():
    """Autocomplete endpoint for movie search"""
    try:
        query = request.args.get('query', '').strip().lower()
        
        if len(query) < 2:
            return jsonify([])
        
        # Filter movies that match the query
        matches = [title for title in movie_titles if query in title.lower()]
        
        # Return top 10 matches
        return jsonify(matches[:10])
        
    except Exception as e:
        logger.error(f"Error in autocomplete: {str(e)}")
        return jsonify([])


if __name__ == '__main__':
    # Load data before starting server
    load_data()
    
    # Run the app
    app.run(debug=True, host='0.0.0.0', port=5000)