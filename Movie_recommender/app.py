import streamlit as st
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# Set page config with enhanced styling
st.set_page_config(
    page_title="CineAI - Movie Recommendation System", 
    page_icon="🎬", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern UI
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    .movie-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
        margin: 1rem 0;
        transition: transform 0.2s ease;
    }
    
    .movie-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 20px rgba(0,0,0,0.15);
    }
    
    .score-badge {
        background: linear-gradient(45deg, #667eea, #764ba2);
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
        margin: 0.5rem 0;
    }
    
    .genre-tag {
        background: #f0f2f6;
        color: #262730;
        padding: 0.2rem 0.6rem;
        border-radius: 15px;
        font-size: 0.8rem;
        margin: 0.2rem;
        display: inline-block;
    }
    
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    .stSelectbox > div > div {
        border-radius: 10px;
    }
    
    .stTextInput > div > div > input {
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Cache data loading and preprocessing
@st.cache_data
def load_and_preprocess_data():
    try:
        # Load data
        movies_data = pd.read_csv('movies.csv')
        ratings_data = pd.read_csv('ratings.csv')
        
        # Clean title function
        def clean_title(title):
            return re.sub("[^a-zA-Z0-9 ]", "", title)
        
        # Enhanced preprocessing for movies data
        movies_data['genres'] = movies_data['genres'].str.split('|')
        movies_data['title'] = movies_data['title'].apply(clean_title)
        movies_data['year'] = movies_data['title'].str.extract(r'\((\d{4})\)')
        movies_data['clean_title'] = movies_data['title'].str.replace(r'\(\d{4}\)', '', regex=True).str.strip()
        
        # Remove movies with no genres
        movies_data = movies_data[~movies_data['genres'].apply(lambda x: '(no genres listed)' in x)]
        
        # Enhanced ratings preprocessing
        ratings_data = ratings_data.drop(['timestamp'], axis=1) if 'timestamp' in ratings_data.columns else ratings_data
        
        # Calculate movie statistics for better recommendations
        movie_stats = ratings_data.groupby('movieId').agg({
            'rating': ['mean', 'count', 'std']
        }).round(2)
        movie_stats.columns = ['avg_rating', 'rating_count', 'rating_std']
        movie_stats = movie_stats.fillna(0)
        
        # Merge everything
        movies_data = movies_data.merge(movie_stats, on='movieId', how='left')
        combined_data = ratings_data.merge(movies_data, on='movieId')
        
        return movies_data, combined_data, ratings_data
        
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None, None, None

@st.cache_data
def setup_enhanced_vectorizers(movies_data):
    # Enhanced title vectorizer with better parameters
    vectorizer_title = TfidfVectorizer(
        ngram_range=(1, 3),
        max_features=5000,
        stop_words='english',
        lowercase=True,
        analyzer='word'
    )
    tfidf_title = vectorizer_title.fit_transform(movies_data['clean_title'].fillna(''))
    
    # Enhanced genre vectorizer
    vectorizer_genre = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=1000
    )
    movies_data['genres_text'] = movies_data['genres'].apply(lambda x: ' '.join(x) if isinstance(x, list) else '')
    tfidf_genre = vectorizer_genre.fit_transform(movies_data['genres_text'])
    
    return vectorizer_title, tfidf_title, vectorizer_genre, tfidf_genre

def enhanced_search_by_title(title, vectorizer_title, tfidf_title, movies_data, top_k=10):
    title = re.sub("[^a-zA-Z0-9 ]", "", title)
    query_vec = vectorizer_title.transform([title])
    similarity = cosine_similarity(query_vec, tfidf_title).flatten()
    
    # Get top matches with minimum similarity threshold
    min_similarity = 0.1
    valid_indices = np.where(similarity >= min_similarity)[0]
    
    if len(valid_indices) == 0:
        return pd.DataFrame()
    
    # Sort by similarity
    sorted_indices = valid_indices[np.argsort(similarity[valid_indices])[::-1]][:top_k]
    results = movies_data.iloc[sorted_indices].copy()
    results['similarity_score'] = similarity[sorted_indices]
    
    return results

def enhanced_scores_calculator(movie_id, combined_data, movies_data, vectorizer_genre, tfidf_genre):
    try:
        # Get users who rated this movie highly (4+ stars)
        similar_users = combined_data.loc[
            (combined_data['movieId'] == movie_id) & (combined_data['rating'] >= 4.0), 
            'userId'
        ].unique()
        
        if len(similar_users) < 5:  # Fallback for movies with few ratings
            similar_users = combined_data.loc[
                (combined_data['movieId'] == movie_id) & (combined_data['rating'] >= 3.5), 
                'userId'
            ].unique()
        
        # Get movies these users liked
        similar_user_recs = combined_data.loc[
            (combined_data['userId'].isin(similar_users)) & 
            (combined_data['rating'] >= 4.0), 
            'movieId'
        ].value_counts(normalize=True)
        
        # Get overall popularity baseline
        all_user_recs = combined_data.loc[
            combined_data['rating'] >= 4.0, 
            'movieId'
        ].value_counts(normalize=True)
        
        # Content-based similarity (genres)
        movie_row = movies_data.loc[movies_data['movieId'] == movie_id]
        if len(movie_row) == 0:
            return pd.DataFrame()
            
        selected_genres = movie_row['genres'].iloc[0]
        if isinstance(selected_genres, list):
            selected_genres = " ".join(selected_genres)
        
        # Find movies with similar genres
        genre_query_vec = vectorizer_genre.transform([selected_genres])
        genre_similarity = cosine_similarity(genre_query_vec, tfidf_genre).flatten()
        
        # Create comprehensive scoring
        scores_data = []
        
        for mid in similar_user_recs.index:
            if mid == movie_id:  # Skip the input movie
                continue
                
            # Collaborative filtering score
            cf_score = similar_user_recs.get(mid, 0)
            popularity_penalty = all_user_recs.get(mid, 0)
            
            # Content-based score
            movie_idx = movies_data.loc[movies_data['movieId'] == mid].index
            if len(movie_idx) > 0:
                cb_score = genre_similarity[movie_idx[0]]
                
                # Movie quality metrics
                movie_info = movies_data.loc[movies_data['movieId'] == mid].iloc[0]
                avg_rating = movie_info.get('avg_rating', 3.0)
                rating_count = movie_info.get('rating_count', 1)
                
                # Rating count penalty for movies with too few ratings
                rating_confidence = min(rating_count / 50, 1.0)
                
                # Combined score with multiple factors
                final_score = (
                    cf_score * 0.4 +  # Collaborative filtering
                    cb_score * 0.3 +  # Content similarity
                    (avg_rating / 5.0) * 0.2 +  # Quality score
                    rating_confidence * 0.1  # Rating reliability
                ) * (1 - popularity_penalty * 0.1)  # Slight penalty for overly popular movies
                
                scores_data.append({
                    'movieId': mid,
                    'score': final_score,
                    'cf_score': cf_score,
                    'cb_score': cb_score,
                    'avg_rating': avg_rating,
                    'rating_count': rating_count
                })
        
        if not scores_data:
            return pd.DataFrame()
            
        scores_df = pd.DataFrame(scores_data)
        return scores_df.sort_values('score', ascending=False)
        
    except Exception as e:
        st.error(f"Error in scoring: {str(e)}")
        return pd.DataFrame()

def get_enhanced_recommendations(user_input, title_idx, vectorizer_title, tfidf_title, vectorizer_genre, tfidf_genre, movies_data, combined_data):
    try:
        title_candidates = enhanced_search_by_title(user_input, vectorizer_title, tfidf_title, movies_data)
        
        if len(title_candidates) <= title_idx:
            return pd.DataFrame()
            
        movie_id = title_candidates.iloc[title_idx]['movieId']
        scores = enhanced_scores_calculator(movie_id, combined_data, movies_data, vectorizer_genre, tfidf_genre)
        
        if scores.empty:
            return pd.DataFrame()
            
        # Merge with movie details
        results = scores.head(15).merge(
            movies_data[['movieId', 'title', 'genres', 'avg_rating', 'rating_count']], 
            on='movieId'
        )[['title', 'score', 'genres', 'avg_rating', 'rating_count']]
        
        return results.head(10)
        
    except Exception as e:
        st.error(f"Error getting recommendations: {str(e)}")
        return pd.DataFrame()

# Enhanced UI Components
def display_movie_card(movie_data, show_score=True):
    with st.container():
        st.markdown('<div class="movie-card">', unsafe_allow_html=True)
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.markdown(f"### 🎬 {movie_data['title']}")
            
            # Genres
            genres_html = ""
            if isinstance(movie_data['genres'], list):
                for genre in movie_data['genres']:
                    genres_html += f'<span class="genre-tag">{genre}</span>'
            st.markdown(genres_html, unsafe_allow_html=True)
            
            # Additional info
            if 'avg_rating' in movie_data and pd.notna(movie_data['avg_rating']):
                st.markdown(f"⭐ **Average Rating:** {movie_data['avg_rating']:.1f}/5.0")
            
            if 'rating_count' in movie_data and pd.notna(movie_data['rating_count']):
                st.markdown(f"📊 **Total Ratings:** {int(movie_data['rating_count'])}")
        
        with col2:
            if show_score and 'score' in movie_data:
                score_percentage = movie_data['score'] * 100
                st.markdown(f'<div class="score-badge">Match: {score_percentage:.1f}%</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🎬 CineAI - Smart Movie Recommendations</h1>
        <p>Discover your next favorite movie with AI-powered recommendations</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load data
    with st.spinner("🔄 Loading movie database..."):
        movies_data, combined_data, ratings_data = load_and_preprocess_data()
    
    if movies_data is None:
        st.error("❌ Could not load movie data. Please ensure 'movies.csv' and 'ratings.csv' are available.")
        st.info("📁 Expected files: movies.csv, ratings.csv")
        return
    
    # Setup vectorizers
    vectorizer_title, tfidf_title, vectorizer_genre, tfidf_genre = setup_enhanced_vectorizers(movies_data)
    
    # Sidebar Navigation
    st.sidebar.markdown("### 🧭 Navigation")
    page = st.sidebar.selectbox(
        "Choose a section:",
        ["🎯 Get Recommendations", "🗂️ Browse by Genre", "📊 Analytics Dashboard"],
        index=0
    )
    
    if page == "🎯 Get Recommendations":
        st.markdown("## 🔍 Find Your Perfect Movie")
        
        # Enhanced search input
        col1, col2 = st.columns([3, 1])
        with col1:
            user_input = st.text_input(
                "🎬 Enter a movie you enjoyed:",
                placeholder="e.g., The Dark Knight, Inception, Avengers...",
                help="Type any movie title to get personalized recommendations"
            )
        
        with col2:
            search_mode = st.selectbox("Search Type:", ["Smart Match", "Exact Match"])
        
        if user_input:
            with st.spinner("🔍 Searching for movies..."):
                title_candidates = enhanced_search_by_title(user_input, vectorizer_title, tfidf_title, movies_data)
            
            if len(title_candidates) > 0:
                st.markdown("### 🎯 Did you mean one of these movies?")
                
                # Display candidates with enhanced UI
                for idx, (_, movie) in enumerate(title_candidates.head(5).iterrows()):
                    with st.expander(f"🎬 {movie['title']} (Match: {movie['similarity_score']*100:.1f}%)", expanded=idx==0):
                        
                        col1, col2, col3 = st.columns([2, 2, 1])
                        
                        with col1:
                            if isinstance(movie['genres'], list):
                                st.markdown("**Genres:** " + ", ".join(movie['genres']))
                            if 'year' in movie and pd.notna(movie['year']):
                                st.markdown(f"**Year:** {movie['year']}")
                        
                        with col2:
                            if 'avg_rating' in movie and pd.notna(movie['avg_rating']):
                                st.metric("Avg Rating", f"{movie['avg_rating']:.1f}/5.0")
                            if 'rating_count' in movie and pd.notna(movie['rating_count']):
                                st.metric("Total Reviews", f"{int(movie['rating_count'])}")
                        
                        with col3:
                            if st.button(f"🚀 Get Recommendations", key=f"rec_{idx}", type="primary"):
                                with st.spinner("🤖 Generating personalized recommendations..."):
                                    recommendations = get_enhanced_recommendations(
                                        user_input, idx, vectorizer_title, tfidf_title, 
                                        vectorizer_genre, tfidf_genre, movies_data, combined_data
                                    )
                                
                                if not recommendations.empty:
                                    st.markdown(f"## 🎬 Movies You'll Love Based on '{movie['title']}'")
                                    
                                    # Display recommendations in a more appealing way
                                    for i, (_, rec) in enumerate(recommendations.iterrows(), 1):
                                        with st.container():
                                            col1, col2 = st.columns([1, 4])
                                            
                                            with col1:
                                                # Rank badge
                                                st.markdown(f"""
                                                <div style="
                                                    background: linear-gradient(45deg, #667eea, #764ba2);
                                                    color: white;
                                                    width: 50px;
                                                    height: 50px;
                                                    border-radius: 50%;
                                                    display: flex;
                                                    align-items: center;
                                                    justify-content: center;
                                                    font-size: 1.2em;
                                                    font-weight: bold;
                                                    margin: auto;
                                                ">#{i}</div>
                                                """, unsafe_allow_html=True)
                                            
                                            with col2:
                                                display_movie_card(rec)
                                            
                                            st.markdown("---")
                                else:
                                    st.warning("😔 No recommendations found for this movie. Try another title!")
            else:
                st.warning("🔍 No movies found matching your search. Please try:")
                st.markdown("""
                - Check the spelling
                - Try a more common movie title  
                - Use fewer words (e.g., "Dark Knight" instead of "The Dark Knight Rises")
                """)
    
    elif page == "🗂️ Browse by Genre":
        st.markdown("## 🎭 Explore Movies by Genre")
        
        # Get all genres
        all_genres = set()
        for genres_list in movies_data['genres'].dropna():
            if isinstance(genres_list, list):
                all_genres.update(genres_list)
        all_genres = sorted(list(all_genres))
        
        col1, col2 = st.columns([2, 1])
        with col1:
            selected_genre = st.selectbox("🎯 Choose your favorite genre:", all_genres)
        with col2:
            sort_by = st.selectbox("Sort by:", ["Rating", "Popularity", "Title"])
        
        if selected_genre:
            # Filter movies by genre
            genre_movies = movies_data[
                movies_data['genres'].apply(
                    lambda x: selected_genre in x if isinstance(x, list) else False
                )
            ].copy()
            
            # Sort based on selection
            if sort_by == "Rating" and 'avg_rating' in genre_movies.columns:
                genre_movies = genre_movies.sort_values(['avg_rating', 'rating_count'], ascending=[False, False])
            elif sort_by == "Popularity" and 'rating_count' in genre_movies.columns:
                genre_movies = genre_movies.sort_values('rating_count', ascending=False)
            else:
                genre_movies = genre_movies.sort_values('title')
            
            st.markdown(f"### 🎬 Top {selected_genre} Movies")
            
            # Display in grid format
            for i in range(0, min(20, len(genre_movies)), 2):
                cols = st.columns(2)
                for j, col in enumerate(cols):
                    if i + j < len(genre_movies):
                        movie = genre_movies.iloc[i + j]
                        with col:
                            display_movie_card(movie, show_score=False)
    
    elif page == "📊 Analytics Dashboard":
        st.markdown("## 📈 Movie Database Insights")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("🎬 Total Movies", f"{len(movies_data):,}")
        
        with col2:
            st.metric("⭐ Total Ratings", f"{len(combined_data):,}")
        
        with col3:
            unique_users = combined_data['userId'].nunique()
            st.metric("👥 Total Users", f"{unique_users:,}")
        
        with col4:
            avg_rating = combined_data['rating'].mean()
            st.metric("📊 Average Rating", f"{avg_rating:.2f}/5.0")
        
        st.markdown("---")
        
        # Visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🎭 Most Popular Genres")
            exploded_data = combined_data.explode('genres')
            genre_counts = exploded_data['genres'].value_counts().head(10)
            
            fig = px.bar(
                x=genre_counts.values,
                y=genre_counts.index,
                orientation='h',
                title="Genre Popularity by Number of Ratings",
                color=genre_counts.values,
                color_continuous_scale="viridis"
            )
            fig.update_layout(showlegend=False, height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### ⭐ Genre Quality Ratings")
            avg_ratings_by_genre = exploded_data.groupby('genres')['rating'].mean().sort_values(ascending=False).head(10)
            
            fig = px.bar(
                x=avg_ratings_by_genre.values,
                y=avg_ratings_by_genre.index,
                orientation='h',
                title="Average Rating by Genre",
                color=avg_ratings_by_genre.values,
                color_continuous_scale="plasma"
            )
            fig.update_layout(showlegend=False, height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        # Rating distribution
        st.markdown("### 📊 Rating Distribution")
        rating_dist = combined_data['rating'].value_counts().sort_index()
        
        fig = px.bar(
            x=rating_dist.index,
            y=rating_dist.values,
            title="Distribution of User Ratings",
            labels={'x': 'Rating', 'y': 'Number of Ratings'},
            color=rating_dist.values,
            color_continuous_scale="blues"
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()