import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.preprocessing import StandardScaler
from scipy.sparse import save_npz, csr_matrix

# Загружаем данные
books = pd.read_csv('books.csv')
rating = pd.read_csv('rating.csv')

print("="*50)
print("1. ПОДГОТОВКА ДАННЫХ ДЛЯ SVD")
print("="*50)

# Проверяем размерность
print(f"Всего оценок: {len(rating)}")
print(f"Уникальных пользователей: {rating['user_id'].nunique()}")
print(f"Уникальных книг: {rating['book_id'].nunique()}")
print(f"Книг в books.csv: {len(books)}")

# Проверяем, все ли book_id из rating есть в books
missing_books = set(rating['book_id']) - set(books['book_id'])
print(f"Книг из rating, которых нет в books: {len(missing_books)}")


print("\n" + "="*50)
print("2. СОЗДАНИЕ МАТРИЦЫ ПОЛЬЗОВАТЕЛЬ-КНИГА")
print("="*50)

# Создаем матрицу (пользователи → строки, книги → столбцы)
user_book_matrix = rating.pivot_table(
    index='user_id',
    columns='book_id',
    values='rating'
).fillna(0)

print(f"Размер матрицы: {user_book_matrix.shape}")
print(f"Плотность: {len(rating) / (user_book_matrix.shape[0] * user_book_matrix.shape[1]) * 100:.2f}%")


print("\n" + "="*50)
print("4. ДОПОЛНИТЕЛЬНЫЕ ПРИЗНАКИ (для гибридной модели)")
print("="*50)

# Признаки книг, которые можно использовать
book_features = books[['book_id', 'average_rating', 'ratings_count']].copy()



scaler = StandardScaler()
book_features['rating_norm'] = scaler.fit_transform(book_features[['average_rating']])
book_features['popularity_norm'] = scaler.fit_transform(book_features[['ratings_count']])

print(book_features.head())


print("\n" + "="*50)
print("5. СОХРАНЕНИЕ ДАННЫХ")
print("="*50)


# Преобразуем в разреженную матрицу (экономия памяти)
sparse_matrix = csr_matrix(user_book_matrix.values)
save_npz('user_book_matrix.npz', sparse_matrix)

# Сохраняем маппинги
user_ids = user_book_matrix.index.tolist()
book_ids = user_book_matrix.columns.tolist()

pd.Series(user_ids).to_csv('user_ids.csv', index=False)
pd.Series(book_ids).to_csv('book_ids.csv', index=False)

# Сохраняем информацию о книгах
book_info = books[['book_id', 'title', 'authors', 'average_rating']].copy()
book_info.to_csv('book_info.csv', index=False)

print("✅ Сохранено:")
print("  - user_book_matrix.npz (матрица оценок)")
print("  - user_ids.csv (список пользователей)")
print("  - book_ids.csv (список книг)")
print("  - book_info.csv (информация о книгах)")

print(f"\n📊 ИТОГОВЫЕ ДАННЫЕ ДЛЯ SVD:")
print(f"  Пользователей: {len(user_ids)}")
print(f"  Книг: {len(book_ids)}")
print(f"  Оценок: {user_book_matrix.values.nonzero()[0].shape[0]}")