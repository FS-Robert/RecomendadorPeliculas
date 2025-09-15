import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

# ==========================
# Cargar datasets
# ==========================
movies = pd.read_csv("movies.csv")
ratings = pd.read_csv("ratings.csv")

# -------------------
# Filtrado colaborativo
# -------------------
ratings_matrix = ratings.pivot_table(index="userId", columns="movieId", values="rating").fillna(0)
collab_sim = cosine_similarity(ratings_matrix.T)
collab_df = pd.DataFrame(collab_sim, index=ratings_matrix.columns, columns=ratings_matrix.columns)

# -------------------
# Filtrado por contenido (géneros + título)
# -------------------
movies["genres"] = movies["genres"].str.replace("|", " ")
movies["content"] = movies["genres"] + " " + movies["title"].str.replace(r'\(\d{4}\)', '', regex=True)
vectorizer = CountVectorizer()
content_matrix = vectorizer.fit_transform(movies["content"])
content_sim = cosine_similarity(content_matrix, content_matrix)
content_df = pd.DataFrame(content_sim, index=movies["movieId"], columns=movies["movieId"])

# ==========================
# Función de recomendación híbrida
# ==========================
def recomendar_peliculas(movie_id, num_recomendaciones=5, alpha=0.5):
    if movie_id not in collab_df.columns or movie_id not in content_df.columns:
        return []

    collab_scores = collab_df[movie_id]
    content_scores = content_df[movie_id]
    combined_scores = alpha * collab_scores + (1 - alpha) * content_scores
    combined_scores = combined_scores.sort_values(ascending=False).iloc[1:num_recomendaciones+1]

    recomendaciones = []
    for pid, score in combined_scores.items():
        titulo = movies[movies['movieId'] == pid]['title'].values[0]
        genres = movies[movies['movieId'] == pid]['genres'].values[0]
        recomendaciones.append((titulo, genres, score))
    return recomendaciones

# ==========================
# Interfaz Streamlit
# ==========================
st.title("🎬 Recomendador de Películas")
st.write("Selecciona una película y obtén recomendaciones combinando ratings y contenido:")

pelicula_seleccionada = st.selectbox("Elige una película:", movies["title"].head(500).values)

if pelicula_seleccionada:
    movie_id = movies[movies['title'] == pelicula_seleccionada]['movieId'].values[0]

    # Verificar si hay datos suficientes
    if movie_id not in collab_df.columns or movie_id not in content_df.columns:
        st.warning("⚠️ No hay suficientes datos para recomendar esta película.")
    else:
        try:
            recomendaciones = recomendar_peliculas(movie_id, num_recomendaciones=5, alpha=0.5)
            if not recomendaciones:
                st.warning("⚠️ No se pudieron generar recomendaciones para esta película.")
            else:
                num_cols = 5
                title_height = 60

                # Mostrar recomendaciones en filas
                for i in range(0, len(recomendaciones), num_cols):
                    row_recs = recomendaciones[i:i+num_cols]
                    cols = st.columns(len(row_recs))
                    for col, (titulo, genres, score) in zip(cols, row_recs):
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
                        col.caption(f"Géneros: {genres} | Similitud: {int(score*100)}%")
        except Exception as e:
            st.error(f"Error al generar recomendaciones: {e}")
