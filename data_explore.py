import pandas as pd
import matplotlib.pyplot as plt


books = pd.read_csv('books.csv')
rating = pd.read_csv('rating.csv')

print(books.sort_values(by='average_rating', ascending=False).head(10)['title'])#топ 10 книг по среднему рейтингу

print(rating['user_id'].value_counts().head(10))#Топ 10 самых активных пользователей

#print(books['ratings_count'].describe())# Посмотри на 25%, 50%, 75% перцентили





plt.subplot(3, 3, 1)
plt.hist(rating['rating'], bins=30, color='blue')
plt.title('Гистограмма рейтингов')
plt.xlabel('Оценка')
plt.ylabel('Количество')
#много оценок 3, 5, особенно 4, но мало оценок 1,2

plt.subplot(3, 3, 2)
plt.hist(books['average_rating'], bins=20, color='green')
plt.title('Гистограмма средних рейтингов')
plt.xlabel('средний рейтинг')
plt.ylabel('Количество')
#схожее с гистограммой рейтингов

plt.subplot(3, 3, 3)
plt.plot(books['average_rating'].sort_values(), books['book_id'], color='red')
plt.title('График отношения среднего рейтинга и id книги')
plt.xlabel("средний рейтинг")
plt.ylabel('id книги')
#получается сигмоида, видимо связано с тем, что изначально книги сортировались по среднему рейтингу, а затем уже выдавали ID

plt.subplot(3, 3, 4)
#Связь между количеством оценок и средним рейтингом
plt.scatter(books['ratings_count'], books['average_rating'], color='black')
plt.title('Связь между количеством оценок и средним рейтингом')
plt.xlabel("количество оценок")
plt.ylabel('средний рейтинг')
plt.xscale('log')


plt.subplot(3, 3, 5)
#Зависимость между рейтингом и годом публикации
plt.hist(books['original_publication_year'], color='brown')
plt.title('Гистограмма годов публикации')
plt.xlabel('Год')
plt.ylabel('Количество')


plt.subplot(3, 3, 6)
#Зависимость между рейтингом и годом публикации
plt.scatter(books['original_publication_year'], books['average_rating'], color='orange')
plt.title('Зависимость между рейтингом и годом публикации')
plt.xlabel("Год публикации")
plt.ylabel('средний рейтинг')


plt.subplot(3, 3, 7)
#Если точки идут по убывающей, то гипотеза подтверждается.
plt.scatter(books['book_id'], books['ratings_count'], color='purple')
plt.xlabel("book_id")
plt.ylabel('ratings_count')



# 1. Группируем книги по автору
author_stats = books.groupby('authors').agg({
    'average_rating': 'mean',
    'title': 'count',
    'ratings_count': 'sum'
}).rename(columns={'title': 'book_count'})

# 2. Смотрим на авторов с >5 книгами
popular_authors = author_stats[author_stats['book_count'] >= 5]

# 3. Топ авторов по среднему рейтингу
print(popular_authors.nlargest(10, 'average_rating'))

# 4. Топ авторов по суммарной популярности (всего оценок)
print(popular_authors.nlargest(10, 'ratings_count'))



# Берем книгу с ID=1 (самая популярная)
book1_ratings = rating[rating['book_id'] == 5]['rating']

# Считаем распределение
print(book1_ratings.value_counts().sort_index())


#plt.show()




