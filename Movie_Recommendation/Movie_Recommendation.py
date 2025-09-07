import numpy as np
import pandas as pd
import difflib # match the name of movie given by user by the movie in the dataset
from sklearn.feature_extraction.text import TfidfVectorizer # convert textual data into numerical values
from sklearn.metrics.pairwise import cosine_similarity # find similarity confidence score of all the movies

movies_data = pd.read_csv("movies.csv")
# print(movies_data.shape)

#select the relevent features for recommendation
selected_features = ['genres','keywords','tagline','cast','director']
# print(selected_features)

#replacing null values with null strings
for feature in selected_features:
    movies_data[feature] = movies_data[feature].fillna('')


# combining all the 5 selected features
combined_features = movies_data['genres']+' '+movies_data['keywords']+' '+movies_data['tagline']+' '+movies_data['cast']+' '+movies_data['director']
# print(combined_features)


# convert the text data into feature vector
vectorizer = TfidfVectorizer()
feature_vectors = vectorizer.fit_transform(combined_features)
# print(feature_vectors)

# getting the similarity scores using cosine similarity
similarity = cosine_similarity(feature_vectors)
# print(similarity)
# print(similarity.shape)

# getting the movie name from the user
movie_name = input(' Enter your favorite movie name : ')

#create a list with all movie names in the dataset
list_of_all_titles = movies_data['title'].tolist()
# print(list_of_all_titles)

# find the close match for movie name given by user
find_close_match = difflib.get_close_matches(movie_name, list_of_all_titles)
# print(find_close_match)
close_match = find_close_match[0]
# print(close_match)

# find the index of the movie with title
index_of_the_movie = movies_data[movies_data.title==close_match]['index'].values[0]
# print(index_of_the_movie)


# getting a list of similar movies
similarity_score = list(enumerate(similarity[index_of_the_movie]))
# print(similarity_score)


# sorting the movies based on their similarity score
sorted_similar_movies = sorted(similarity_score, key = lambda x:x[1], reverse = True) 
# print(sorted_similar_movies)


# print the name of similar movies based on the index
print('Movies suggested for you : \n')
i = 1
for movie in sorted_similar_movies:
  index = movie[0]
  title_from_index = movies_data[movies_data.index==index]['title'].values[0]
  if (i<31):
    print(i, '.',title_from_index)
    i+=1