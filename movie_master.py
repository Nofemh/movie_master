import streamlit as st
import requests
import random
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

API_KEY = 'e78b1ee71e70e303478ef6ba5e29e2e9'  # TMDb API Key
OMDB_API_KEY = '8ad270f7'  # OMDb API Key
BASE_URL = 'https://api.themoviedb.org/3/'

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
ps = PorterStemmer()

def fetch_movies(endpoint, params):
    movies = []
    for page in range(1, 11):  # Fetching 10 pages as an example
        params['page'] = page
        response = requests.get(BASE_URL + endpoint, params=params)
        data = response.json()
        movies.extend(data['results'])
    return movies

def fetch_top_movies():
    return fetch_movies('movie/top_rated', {'api_key': API_KEY})

def preprocess_text(text):
    tokens = text.split()
    tokens = [token.lower() for token in tokens if token.isalpha()]
    tokens = [token for token in tokens if token not in stop_words]
    tokens = [ps.stem(token) for token in tokens]
    return ' '.join(tokens)

def preprocess_data(movies):
    df = pd.DataFrame(movies)
    df['text'] = df['title'].fillna('') + " " + df['overview'].fillna('')
    df['text'] = df['text'].apply(preprocess_text)
    return df

def similar_story(query, df, num_results=10):
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(df['text'])
    
    query_vec = vectorizer.transform([preprocess_text(query)])
    similarity = cosine_similarity(query_vec, tfidf_matrix).flatten()
    
    top_indices = similarity.argsort()[-num_results:][::-1]  # Get top N similar movies
    return df.iloc[top_indices]

def search_movies_by_genre(genre):
    endpoint = 'discover/movie'
    params = {
        'api_key': API_KEY,
        'with_genres': genre
    }
    response = requests.get(BASE_URL + endpoint, params=params)
    return response.json()

def surprise_me():
    endpoint = 'discover/movie'
    params = {
        'api_key': API_KEY,
        'sort_by': 'popularity.desc',
        'page': random.randint(1, 100)  # Random page to get a random movie
    }
    response = requests.get(BASE_URL + endpoint, params=params)
    results = response.json()
    return results['results'][0] if 'results' in results else None

def fetch_movie_details_from_omdb(title):
    url = f"http://www.omdbapi.com/"
    params = {
        't': title,
        'apikey': OMDB_API_KEY
    }
    response = requests.get(url, params=params)
    return response.json()

def display_movie(movie):
    col1, col2 = st.columns([1, 4])  # Adjust column ratios as needed
    
    with col1:
        if movie.get('poster_path'):
            poster_url = f"https://image.tmdb.org/t/p/w500{movie['poster_path']}"
            st.image(poster_url, use_column_width=True, caption="Poster")
    
    with col2:
        st.write(f"**Title**: {movie['title']}")
        st.write(f"**Overview**: {movie['overview']}")
        
        # Fetch parental guide info from OMDb
        omdb_data = fetch_movie_details_from_omdb(movie['title'])
        parental_guide = omdb_data.get('Rated', 'Not Available')

        st.write(f"**Parental Guide**: {parental_guide}")
    
    st.write("---")

def main():
    st.title("🎬 Movie Suggestion App 🎬")

    st.sidebar.title("Options")
    options = {
        'Genre': '🎭 Choose a Genre',
        'Similar Story': '🔍 Find Similar Stories',
        'Surprise Me': '🎲 Surprise Me!',
        'Current Top IMDb': '⭐ Current Top IMDb Movies'
    }

    selected_option = st.sidebar.radio("Select an option", list(options.values()))

    if selected_option == options['Genre']:
        genre_options = [
            'Select an option', 'Action', 'Adventure', 'Animation', 'Comedy',
            'Crime', 'Documentary', 'Drama', 'Family', 'Fantasy', 'History',
            'Horror', 'Music', 'Mystery', 'Romance', 'Science Fiction', 
            'TV Movie', 'Thriller', 'War', 'Western'
        ]
        selected_genre = st.selectbox("Choose a genre", genre_options)

        if selected_genre != 'Select an option':
            genre_ids = {
                'Action': 28,
                'Adventure': 12,
                'Animation': 16,
                'Comedy': 35,
                'Crime': 80,
                'Documentary': 99,
                'Drama': 18,
                'Family': 10751,
                'Fantasy': 14,
                'History': 36,
                'Horror': 27,
                'Music': 10402,
                'Mystery': 9648,
                'Romance': 10749,
                'Science Fiction': 878,
                'TV Movie': 10770,
                'Thriller': 53,
                'War': 10752,
                'Western': 37
            }

            genre_id = genre_ids[selected_genre]

            if st.button("Submit"):
                results = search_movies_by_genre(genre_id)

                if 'results' in results:
                    st.header("Movie Suggestions:")
                    for movie in results['results']:
                        display_movie(movie)

    elif selected_option == options['Similar Story']:
        user_query = st.text_input("Enter keywords to describe the movie you have in mind:")
        if st.button("Submit"):
            movies = fetch_movies('discover/movie', {'api_key': API_KEY})
            df = preprocess_data(movies)
            results = similar_story(user_query, df)

            st.header("Movie Suggestions:")
            for _, movie in results.iterrows():
                display_movie(movie)

    elif selected_option == options['Surprise Me']:
        if st.button("Get Surprise"):
            movie = surprise_me()
            if movie:
                st.header("Random Movie Suggestion:")
                display_movie(movie)

    elif selected_option == options['Current Top IMDb']:
        if st.button("Show Top IMDb"):
            top_movies = fetch_top_movies()

            st.header("Current Top IMDb Movies:")
            for movie in top_movies:
                display_movie(movie)

if __name__ == "__main__":
    main()


