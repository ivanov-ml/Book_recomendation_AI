import pandas as pd
import urllib.request
import ssl

# Отключаем проверку SSL (для этого запроса)
ssl._create_default_https_context = ssl._create_unverified_context

url = "https://raw.githubusercontent.com/zygmuntz/goodbooks-10k/master/books.csv"
books = pd.read_csv(url)#скачанные данные по книгам

url_ratings = "https://raw.githubusercontent.com/zygmuntz/goodbooks-10k/master/ratings.csv"
ratings = pd.read_csv(url_ratings)#скачанные данные по рейтингам

books_cleaned = books[[
    'book_id',           # уникальный ID книги (главный ключ!)
    'title',             # название
    'authors',           # автор (пригодится для базовых рекомендаций)
    'average_rating',    # средний рейтинг
    'original_publication_year',# оригинальный год публикации
    'ratings_count'      # количество оценок
]]

books_cleaned.to_csv("books.csv", index=False)
ratings.to_csv("rating.csv", index=False)
