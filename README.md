# Movie Recommendation System

A content-based movie recommendation system built with Flask, scikit-learn, and TMDb API. The system uses TF-IDF vectorization and 
cosine similarity to recommend movies based on metadata like genres, cast, director, and keywords.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Flask](https://img.shields.io/badge/Flask-3.0.0-green)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3.0-orange)

## Features

- Content-based filtering using TF-IDF and Cosine Similarity
- Real-time movie poster fetching from TMDb API
- Fuzzy string matching for handling typos
- Autocomplete search functionality
- Responsive web interface
- Pre-computed similarity matrix for fast recommendations



## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. Clone the repository
```bash
git clone https://github.com/yourusername/movie-recommendation-system.git
cd movie-recommendation-system
```

2. Install required packages
```bash
pip install -r requirements.txt
```

3. Get TMDb API Key
   - Sign up at [TMDb](https://www.themoviedb.org/signup)
   - Request an API key from [API Settings](https://www.themoviedb.org/settings/api)
   - Copy your API Key (v3 auth)

4. Configure API Key
   - Open `app.py`
   - Replace `YOUR_TMDB_API_KEY_HERE` with your actual API key

5. Run the application
```bash
python app.py
```


## Dataset

The system uses a movie dataset (`movies.csv`) with the following columns:
- `title`: Movie title
- `genres`: Movie genres
- `keywords`: Associated keywords
- `tagline`: Movie tagline
- `cast`: Cast members
- `director`: Director name
- `index`: Unique identifier

Dataset should contain at least these columns for the recommendation algorithm to work properly.

## Project Structure

```
movie-recommendation-system/
│
├── app.py                  # Flask application
├── movies.csv             # Movie dataset
├── requirements.txt       # Python dependencies
├── templates/
│   └── index.html        # Frontend interface
└── README.md             # Project documentation
```

## Usage

1. Enter a movie name in the search box
2. Select from autocomplete suggestions (optional)
3. Adjust the number of recommendations (5-50)
4. Click "Get Recommendations"
5. View recommended movies with posters and similarity scores

## Technologies Used

- **Backend**: Flask (Python web framework)
- **Machine Learning**: scikit-learn (TF-IDF, Cosine Similarity)
- **API**: TMDb API (Movie posters and metadata)
- **Frontend**: HTML5, CSS3, JavaScript
- **Libraries**: pandas, numpy, difflib

## How It Works

1. **Data Preprocessing**: Movie metadata (genres, keywords, tagline, cast, director) is combined into a single feature vector
2. **Vectorization**: TF-IDF vectorizer converts text data into numerical vectors
3. **Similarity Calculation**: Cosine similarity is computed between all movie pairs
4. **Recommendation**: When a user searches for a movie, the system finds the most similar movies based on pre-computed similarity scores
5. **Poster Fetching**: Movie posters are dynamically fetched from TMDb API with caching

## API Endpoints

### `GET /`
Returns the main application page

### `POST /predict`
Get movie recommendations

**Request Body:**
```json
{
  "movie_name": "Iron Man",
  "top_n": 20
}
```

**Response:**
```json
{
  "status": "success",
  "matched_movie": "Iron Man",
  "matched_poster": "https://image.tmdb.org/t/p/w500/...",
  "recommendations": [
    {
      "rank": 1,
      "title": "Iron Man 2",
      "similarity_score": 0.4095,
      "poster_url": "https://image.tmdb.org/t/p/w500/..."
    }
  ]
}
```

### `GET /autocomplete?query=<search_term>`
Get movie title suggestions

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [TMDb API](https://www.themoviedb.org/documentation/api) for movie data and posters
- [scikit-learn](https://scikit-learn.org/) for machine learning algorithms
- Movie dataset from [Kaggle](https://www.kaggle.com/)

