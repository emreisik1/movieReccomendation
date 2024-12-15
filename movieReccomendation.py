import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Filmler için bir içerik tabanlı öneri sistemi
class MovieRecommender:
    def __init__(self, movies_data_path):
        # Filmleri yükleme
        self.movies_data = pd.read_csv(movies_data_path)
        self.movies_data['genres'] = self.movies_data['genres'].fillna('')

        # TF-IDF vektörleştirici
        self.vectorizer = TfidfVectorizer(tokenizer=lambda x: x.split('|'), lowercase=False)
        self.genre_matrix = self.vectorizer.fit_transform(self.movies_data['genres'])

        # Kullanıcı tercihleri
        self.liked_movies = pd.DataFrame(columns=self.movies_data.columns)
        self.disliked_movies = pd.DataFrame(columns=self.movies_data.columns)

    def update_preferences(self, movie_title, liked):
        # Kullanıcı tercihlerine göre sevilen veya sevilmeyen filmleri güncelle
        movie = self.movies_data[self.movies_data['title'] == movie_title]
        if movie.empty:
            print("Hata: Film veri setinde bulunamadı.")
            return

        if liked:
            self.liked_movies = pd.concat([self.liked_movies, movie], ignore_index=True)
            print(f"{movie_title} sevdiğiniz filmler listesine eklendi.")
        else:
            self.disliked_movies = pd.concat([self.disliked_movies, movie], ignore_index=True)
            print(f"{movie_title} sevmediğiniz filmler listesine eklendi.")

    def recommend_movies(self, top_n=3):
        if self.liked_movies.empty:
            print("Henüz sevdiğiniz film bulunmuyor. Lütfen önce sevdiğiniz filmleri belirtin.")
            return

        # Sevilen filmlerin tür matrisinin ortalamasını alarak kullanıcı profili oluştur
        liked_indices = self.movies_data[self.movies_data['title'].isin(self.liked_movies['title'])].index
        user_profile = np.asarray(self.genre_matrix[liked_indices].mean(axis=0))

        # Kullanıcı profiline göre tüm filmlerin benzerliğini hesapla
        similarities = cosine_similarity(user_profile, self.genre_matrix).flatten()

        # Sevilen ve sevilmeyen filmleri önerilerden çıkar
        self.movies_data['similarity'] = similarities
        recommended_movies = self.movies_data[
            ~self.movies_data['title'].isin(self.liked_movies['title']) &
            ~self.movies_data['title'].isin(self.disliked_movies['title'])
        ]

        # Benzerliğe göre sıralayıp en iyi önerileri döndür
        top_recommendations = recommended_movies.sort_values(by='similarity', ascending=False).head(top_n)

        print("\nÖnerilen Filmler:")
        for idx, row in top_recommendations.iterrows():
            print(f"- {row['title']} (Benzerlik: %{row['similarity'] * 100:.2f})")

    def start_system(self):
        print("Film Öneri Sistemine Hoş Geldiniz!")

        while True:
            print("\n1. Sevdiğiniz bir filmi ekleyin")
            print("2. Sevmediğiniz bir filmi ekleyin")
            print("3. Film önerisi alın")
            print("4. Çıkış")

            choice = input("Seçiminiz: ").strip()

            if choice == '1':
                movie_title = input("Sevdiğiniz film adı: ").strip()
                self.update_preferences(movie_title, liked=True)
            elif choice == '2':
                movie_title = input("Sevmediğiniz film adı: ").strip()
                self.update_preferences(movie_title, liked=False)
            elif choice == '3':
                self.recommend_movies()
            elif choice == '4':
                print("Sistemden çıkılıyor. Teşekkürler!")
                break
            else:
                print("Geçersiz seçim. Lütfen tekrar deneyin.")

# Ana program
if __name__ == "__main__":
    recommender = MovieRecommender("C:/Users/emrei/Desktop/movieRecco/movies.csv")
    recommender.start_system()
