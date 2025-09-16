import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
import requests
import re

# ==========================
# Configuración OMDb
# ==========================
API_KEY = "20a69d41"

@st.cache_data
def obtener_poster(titulo):
    """Obtiene el poster de una película desde OMDb, maneja errores."""
    try:
        titulo_limpio = re.sub(r'\(\d{4}\)', '', titulo).strip()
        url = f"http://www.omdbapi.com/?t={titulo_limpio}&apikey={API_KEY}"
        response = requests.get(url, timeout=5).json()
        if response.get("Poster") and response["Poster"] != "N/A":
            return response["Poster"]
    except Exception as e:
        st.warning(f"No se pudo obtener poster para {titulo}: {e}")
    return None

# ==========================
# Cargar datasets
# ==========================
movies = pd.read_csv("movies.csv")
ratings = pd.read_csv("ratings.csv")
movies = movies.head(500)
ratings = ratings[ratings['movieId'].isin(movies['movieId'])]

# Limpiar títulos
movies['clean_title'] = movies['title'].str.replace(r'\(\d{4}\)', '', regex=True).str.strip()

# ==========================
# Filtrado colaborativo
# ==========================
ratings_matrix = ratings.pivot_table(index="userId", columns="movieId", values="rating").fillna(0)
collab_sim = cosine_similarity(ratings_matrix.T)
collab_df = pd.DataFrame(collab_sim, index=ratings_matrix.columns, columns=ratings_matrix.columns)

# ==========================
# Filtrado por contenido
# ==========================
movies["genres"] = movies["genres"].str.replace("|", " ")
movies["content"] = movies["genres"] + " " + movies["clean_title"]
vectorizer = CountVectorizer()
content_matrix = vectorizer.fit_transform(movies["content"])
content_sim = cosine_similarity(content_matrix, content_matrix)
content_df = pd.DataFrame(content_sim, index=movies["movieId"], columns=movies["movieId"])

# ==========================
# Función de recomendación híbrida
# ==========================
def recomendar_peliculas(movie_id, num_recomendaciones=5, alpha=0.5):
    """Devuelve una lista de películas recomendadas con posters y puntuación."""
    try:
        if movie_id not in collab_df or movie_id not in content_df:
            return []

        collab_scores = collab_df[movie_id]
        content_scores = content_df[movie_id]
        combined_scores = alpha * collab_scores + (1 - alpha) * content_scores
        combined_scores = combined_scores.sort_values(ascending=False).iloc[1:num_recomendaciones+1]

        recomendaciones = []
        for pid, score in combined_scores.items():
            try:
                titulo = movies.loc[movies['movieId'] == pid, 'title'].values[0]
                genres = movies.loc[movies['movieId'] == pid, 'genres'].values[0]
                poster = obtener_poster(titulo)
                recomendaciones.append((titulo, genres, poster, score))
            except IndexError:
                continue  # Saltar si el movieId no existe
        return recomendaciones
    except Exception as e:
        st.error(f"Error al generar recomendaciones: {e}")
        return []

# ==========================
# Interfaz Streamlit
# ==========================
st.title("🎬 Recomendador de Películas")
st.write("Selecciona una película y obtén recomendaciones combinando ratings y contenido:")

pelicula_seleccionada = st.selectbox("Elige una película:", movies["clean_title"].head(500).values)

if pelicula_seleccionada:
    try:
        movie_id = movies.loc[movies['clean_title'] == pelicula_seleccionada, 'movieId'].iloc[0]
        recomendaciones = recomendar_peliculas(movie_id, num_recomendaciones=5, alpha=0.5)

        if not recomendaciones:
            st.warning("No se encontraron recomendaciones para esta película.")
        else:
            # Mostrar recomendaciones en columnas
            num_cols = 5
            title_height = 60
            for i in range(0, len(recomendaciones), num_cols):
                row_recs = recomendaciones[i:i+num_cols]
                cols = st.columns(len(row_recs))
                for col, (titulo, _, poster, score) in zip(cols, row_recs):
                    col.markdown(
                        f"""
                        <div style="
                            height:{title_height}px;
                            overflow-y:auto;
                            text-align:center;
                            font-weight:bold;
                            font-size:16px;
                            margin-bottom:5px;">
                            {titulo}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                    col.caption(f"Similitud: {int(score*100)}%")
                    if poster:
                        col.image(poster, width='stretch')  # Sustituye use_container_width
                    else:
                        col.write("❌ No hay ningún poster disponible")
    except IndexError:
        st.error("No se encontró la película seleccionada. Intenta otra.")
    except Exception as e:
        st.error(f"Ocurrió un error inesperado: {e}")
