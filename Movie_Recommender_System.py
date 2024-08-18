import numpy as np
import pandas as pd
import ast
import pickle
from sklearn.feature_extraction.text import CountVectorizer 
from nltk.stem.porter import PorterStemmer
from sklearn.metrics.pairwise import cosine_similarity

movies = pd.read_csv("C:\\Users\\HP\\Downloads\\tmdb_5000_movies.csv\\tmdb_5000_movies.csv")
credits = pd.read_csv("C:\\Users\\HP\\Downloads\\tmdb_5000_credits.csv\\tmdb_5000_credits.csv")
data = movies.merge(credits,on='title')

# genres, id, keywords, title, overview, cast, crew
data = data[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]
data.dropna(inplace=True)
#print(data.duplicated().sum())
#print(data.iloc[0].genres)
def convert(obj):
    l=[]
    for i in ast.literal_eval(obj):
        l.append(i['name'])
    return l

data['genres'] = data['genres'].apply(convert)

data['keywords'] = data['keywords'].apply(convert)

#print(data.iloc[0].cast)

def convert3(obj):
    l=[]
    counter=0
    for i in ast.literal_eval(obj):
        if counter!=3:
            l.append(i['name'])
            counter+=1
        else:
            break
    return l

data['cast'] = data['cast'].apply(convert3)

#print(data['crew'][0])
def fetch_director(obj):
    l=[]
    for i in ast.literal_eval(obj):
        if i['job'] == 'Director':
            l.append(i['name'])
            break
    return l

data['crew'] = data['crew'].apply(fetch_director)

#print(data['overview'][0])
data['overview'] = data['overview'].apply(lambda x:x.split())
#print(data['overview'][0])

data['genres'] = data['genres'].apply(lambda x:[i.replace(' ','') for i in x])
data['keywords'] = data['keywords'].apply(lambda x:[i.replace(' ','') for i in x])
data['cast'] = data['cast'].apply(lambda x:[i.replace(' ','') for i in x])
data['crew'] = data['crew'].apply(lambda x:[i.replace(' ','') for i in x])

data['tags'] = data['overview'] + data['genres'] + data['keywords'] + data['cast'] + data['crew']

new_df = data[['movie_id','title','tags']]

new_df['tags'] = new_df['tags'].apply(lambda x:" ".join(x))

new_df['tags'] = new_df['tags'].apply(lambda x:x.lower())

cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(new_df['tags'].to_numpy())
#print(vectors.shape)
#print(vectors.toarray()[0])

ps = PorterStemmer()
def stem(text):
    y=[]
    for i in text.split():
        y.append(ps.stem(i))

    string = " ".join(y)
    return string

new_df['tags'] = new_df['tags'].apply(stem)
#print(cv.get_feature_names_out())

similarity = cosine_similarity(vectors)

def recommend(movie):
    movie_index = new_df[new_df['title'] == movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x:x[1])[1:6]
    for i in movies_list:
        print(new_df.iloc[i[0]].title)

print(recommend('Batman Begins'))

pickle.dump(new_df.to_dict(),open('movie_dict.pkl','wb'))
pickle.dump(similarity,open('similarity.pkl','wb'))