#các thư viện cần dùng
import flask
from flask import Flask, render_template, request, session, redirect, url_for
import csv
import difflib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics import mean_squared_error
import math

app = flask.Flask(__name__, template_folder='templates')
app.secret_key = 'somesecretkeythatonlyishouldknow'

#đọc dataset
df2 = pd.read_csv('./model/tmdb1.csv', sep=',', dtype={'a': str})
df = pd.read_csv('./model/poster_data.csv')

#lấy các trường : poster, title, id, detail
all_poster = [df2['poster'][i] for i in range(len(df2['poster']))]
all_titles = [df2['title'][i] for i in range(len(df2['title']))]
all_ids = [df2['stt'][i] for i in range(len(df2['stt']))]
all_details = [df2['homepage'][i] for i in range(len(df2['homepage']))]

#tạo 1 matrix, stop_words=english vì trường dữ liệu để fit: soup = tiếng Anh
count = CountVectorizer(stop_words='english')
#fit and transform over the SOUP column => count_matrix
count_matrix = count.fit_transform(df2['soup'].values.astype('U'))
#consine_similarity: xac dinh độ tương tự của count_matrix
cosine_sim2 = cosine_similarity(count_matrix, count_matrix)

df2 = df2.reset_index()
#tìm kiếm sự xuất của dữ liệu hiện trong dataset.
indices1 = pd.Series(df2.index, index=df2['title'])
indices2 = pd.Series(df2.index, index=df2['genres'])
indices3 = pd.Series(df2.index, index=df2['director'])
indices4 = pd.Series(df2.index, index=df2['cast'])

#user object
class User:
    def __init__(self, id, username, password):
        self.id = id
        self.username = username
        self.password = password
    def __repr__(self):
        return f'<User: {self.id}>'
# 1 user to append
users = []
users.append(User(id=100, username='NguyenNgocPhong', password='1'))

#movie parse on screen
class Movie:
    def __init__(self, title, poster, ids, detail):
        self.title = title
        self.poster = poster
        self.ids = ids
        self.detail = detail
movies = []
for i in range(len(all_titles)):
    movies.append(Movie(title=all_titles[i], poster=all_poster[i], ids=all_ids[i], detail = all_details[i]))



#recommend from survay
def get_recommendations(title, genres, director, cast):
    #xac dinh do tuong tu cua 2 matrix.
    cosine_sim = cosine_similarity(count_matrix, count_matrix)
    #tim` su xuat hien cua du lieu.
    idx1 = indices1[title]
    idx2 = indices2[genres]
    idx3 = indices3[director]
    idx4 = indices4[cast]
    
    #ma trận độ tương tự với từng trường(1: title, 2: director, 3:genres, 4: cast)
    sim_scores1 = list(enumerate(cosine_sim[idx1]))
    sim_scores2 = list(enumerate(cosine_sim[idx2][0]))
    sim_scores3 = list(enumerate(cosine_sim[idx3][0]))
    sim_scores4 = list(enumerate(cosine_sim[idx4]))
    
    #sort matran tren
    sim_scores1 = sorted(sim_scores1, key=lambda x: x[1], reverse=True)
    sim_scores2 = sorted(sim_scores2, key=lambda x: x[1], reverse=True)
    sim_scores3 = sorted(sim_scores3, key=lambda x: x[1], reverse=True)
    sim_scores4 = sorted(sim_scores4, key=lambda x: x[1], reverse=True)
    
    #get top 10, 4 cai la 40 film.
    sim_scores1 = sim_scores1[1:11]
    sim_scores2 = sim_scores2[1:11]
    sim_scores3 = sim_scores3[1:11]
    sim_scores4 = sim_scores4[1:11]
    
    # tu` top 40, lay ra title, detail, .. (cac thuoc tinh)
    tit = []
    movie_indices = [i[0] for i in sim_scores1]
    tit1 = df2['title'].iloc[movie_indices]

    movie_indices = [i[0] for i in sim_scores2]
    tit2 = df2['title'].iloc[movie_indices]

    movie_indices = [i[0] for i in sim_scores3]
    tit3 = df2['title'].iloc[movie_indices]

    movie_indices = [i[0] for i in sim_scores4]
    tit4 = df2['title'].iloc[movie_indices]
    
    tit = tit1.append(tit2.append(tit3.append(tit4)))
    
    detail = df2['homepage'].iloc[movie_indices]
    ids = df2['stt'].iloc[movie_indices]
    posters = df2['poster'].iloc[movie_indices]
    return_df = pd.DataFrame(columns=['Title', 'Detail', 'ID', 'Poster'])
    return_df['Title'] = tit
    return_df['Detail'] = detail
    return_df['ID'] = ids
    return_df['Poster'] = posters
    #return 1 DataFrame gom` 4 truong`.
    return return_df

#dat ten cho columns
rs_cols = ['user_id', 'movie_id', 'rating']
#get du lieu
ratings_base = pd.read_csv('model/rate_train.csv', names=rs_cols, encoding='latin-1')
ratings_test = pd.read_csv('model/rate_test.csv', names=rs_cols, encoding='latin-1')

#xac dinh số lượng user, film trong tập train và test.
n_users_base = ratings_base['user_id'].unique().max()
n_items_base = ratings_base['movie_id'].unique().max()
n_users_test = ratings_test['user_id'].unique().max()
n_items_test = ratings_test['movie_id'].unique().max()

#tạo ma trận user-item, được lấp đầy = lượt rate, nếu user chưa rate 1 bố phim => 0
train_matrix = np.zeros((n_users_base, n_items_base))
for line in ratings_base.itertuples():
    train_matrix[line[1]-1,line[2]-1] = line[3]
test_matrix = np.zeros((n_users_test, n_items_test))
for line in ratings_test.itertuples():
    test_matrix[line[1]-1,line[2]-1] = line[3]
#item-item based collaborative filtering   
item_similarity = pairwise_distances(train_matrix.T, metric = 'cosine')
#user-user based collaborative filtering   
user_similarity = pairwise_distances(train_matrix, metric='cosine')

#item-item matrix
def predict_item_item(train_matrix, item_similarity, n_similar=30):
    similar_n = item_similarity.argsort()[:,-n_similar:][:,::-1]
    print('similar_n shape: ', similar_n.shape)
    pred = np.zeros((n_users_base,n_items_base))
    for i,items in enumerate(similar_n):
        similar_items_indexes = items
        similarity_n = item_similarity[i,similar_items_indexes]
        matrix_n = train_matrix[:,similar_items_indexes]
        rated_items = matrix_n.dot(similarity_n)/similarity_n.sum()
        pred[:,i]  = rated_items
    return pred
#user-user matrix
def predict_user_user(train_matrix, user_similarity, n_similar = 30):
    
    similar_n = user_similarity.argsort()[:,-n_similar:][:,::-1]
    pred = np.zeros((n_users_base,n_items_base))
    
    for i,users in enumerate(similar_n):
        similar_users_indexes = users
        similarity_n = user_similarity[i,similar_users_indexes]
        matrix_n = train_matrix[similar_users_indexes,:]
        rated_items = similarity_n[:,np.newaxis].T.dot(matrix_n - matrix_n.mean(axis=1)[:,np.newaxis])/ similarity_n.sum()
        pred[i,:]  = rated_items
        
    return pred

#recommend from user
def  get_user_recommendations(user_id):
    predictions = predict_user_user(train_matrix,user_similarity, 50) + train_matrix.mean(axis=1)[:, np.newaxis]
    #predicted_ratings = predictions[test_matrix.nonzero()]
    #test_truth = test_matrix[test_matrix.nonzero()]
    user_ratings = predictions[user_id-1,:]
    train_unkown_indices = np.where(train_matrix[user_id-1,:] == 0)[0]
    user_recommendations = user_ratings[train_unkown_indices]
    movie = []
    for movie_id in user_recommendations.argsort()[-6:][: : -1]:
        movie.append(movie_id +1)
    tit = df2['title'].iloc[movie]
    detail = df2['homepage'].iloc[movie]
    ids = df2['stt'].iloc[movie]
    posters = df2['poster'].iloc[movie]
    return_df = pd.DataFrame(columns=['Title', 'Detail', 'ID', 'Poster'])
    return_df['Title'] = tit
    return_df['Detail'] = detail
    return_df['ID'] = ids
    return_df['Poster'] = posters
    return return_df
    
#write to csv
def writer_csv(user_id, movie_id):
    file_name = "./model/rate_train.csv"
    rows = [
        [user_id, movie_id, "5"]
    ]
    with open(file_name,'a', newline='') as csvfile: 
        
        # creating a csv writer object  
        csvwriter = csv.writer(csvfile)  
        # writing the data rows  
        csvwriter.writerows(rows) 

#app routes 
@app.route('/', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        session.pop('user_id', None)

        username = flask.request.form['id']
        password = flask.request.form['password']
        user = [x for x in users if x.username == username][0]
        if user and user.password == password:
            session['user_id'] = user.id
            return redirect(url_for('filtering', user_id = user.id, user_name = user.username))
        return redirect(url_for('login'))
    return flask.render_template("login.html")

@app.route('/user_home', methods = ['POST', 'GET'])
def filtering():
    if request.method == 'GET':
        user_id = request.args['user_id']
        user_name = request.args['user_name']
        user_id = session['user_id']
        result = get_user_recommendations(user_id)
        names = []
        details = []
        ids = []
        posters = []
        movies_recommend = []
        for i in range(len(result)):
            names.append(result.iloc[i][0])
            details.append(result.iloc[i][1])
            ids.append(result.iloc[i][2])
            posters.append(result.iloc[i][3])
        for i in range(len(result)):
            movies_recommend.append(Movie(title=names[i], detail = details[i], poster = posters[i], ids= ids[i]))
        return flask.render_template("user_home.html", movies = movies, movies_recommend = movies_recommend, user_id = user_id, user_name = user_name)
    if request.method == 'POST':
        user_id = request.args['user_id']
        user_name = request.args['user_name']
        movie_id = request.form['btn_care']
        writer_csv(user_id, movie_id)
        return redirect(url_for('filtering', user_id = user_id, user_name = user_name))


@app.route('/survay')
def survay():
    return flask.render_template("index.html", movies = movies)

@app.route('/home')
def home():
    return flask.render_template("home.html", movies = movies)

@app.route('/interest', methods = ['GET', 'POST'])
def interest():
    if flask.request.method == 'POST':
        m_name = flask.request.form['movie_name']
        m_director = flask.request.form['director']
        m_genres = flask.request.form['genres']
        m_cast = flask.request.form['cast']
        m_name = m_name.title()
        #check = difflib.get_close_matches(m_name,all_titles,cutout=0.50,n=1)
        result_final = get_recommendations(m_name, m_genres, m_director, m_cast)
        names = []
        details = []
        ids = []
        posters = []
        moviesX = []
        for i in range(len(result_final)):
            names.append(result_final.iloc[i][0])
            details.append(result_final.iloc[i][1])
            ids.append(result_final.iloc[i][2])
            posters.append(result_final.iloc[i][3])
        for i in range(len(result_final)):
            moviesX.append(Movie(title=names[i], detail = details[i], poster = posters[i], ids= ids[i]))
        return flask.render_template("positive.html", movie_names=moviesX, search_name=m_name)

if __name__ == '__main__':
    app.run(debug = True)