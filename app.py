
import pandas as pd
import pickle
import statsmodels.api as sm
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import seaborn as sns
from sklearn.preprocessing import LabelEncoder       # instantiate labelencoder object
from sklearn.preprocessing import StandardScaler

from sklearn.feature_extraction import DictVectorizer 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import SelectKBest   #feature selection best features
from sklearn.feature_selection import chi2         # clase chi-cuadrado 
from sklearn.feature_selection import f_classif, f_regression    #clase f_clasificador ANOVA
from sklearn.feature_selection import RFE, SelectFromModel

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RidgeCV, LassoCV, Ridge, Lasso
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.neural_network import MLPClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.svm import LinearSVC, SVR

import joblib

from sklearn.metrics import r2_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn import metrics 

from flask import Flask, Response, render_template, flash, request, redirect, url_for, session, jsonify, send_file, send_from_directory
from werkzeug.utils import secure_filename
from flask_cors import CORS, cross_origin
import logging, os, datetime, subprocess
import urllib.request
import io, base64, json

from werkzeug.local import LocalProxy
from collections import Counter
from joblib import dump, load
from datetime import datetime
from sqlalchemy import create_engine

ALLOWED_EXTENSIONS = set(['txt', 'csv', 'png', 'jpg', 'jpeg', 'gif'])
UPLOAD_FOLDER = 'data/'

app = Flask(__name__)
CORS(app)
app.config.from_object(__name__)

app.secret_key = os.urandom(24)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

logging.basicConfig(level=logging.DEBUG) # logger config in development mode

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
 return render_template('home.html')

@app.route('/prediction', methods=['POST'])
def predict():
    model = request.form.get('modelo')
    if 'file' not in request.files:
        #flash('No file part')
        #return redirect('http://localhost:8080')
        resp = jsonify({'message' : 'No file part in the request'})
        resp.status_code = 400
        return resp
    file = request.files['file']
    if file.filename == '':
        resp = jsonify({'message' : 'No hay file seleccionada para cargar'})
        resp.status_code = 400
        return resp
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join('new/', filename))
        
        urldata = 'new/'+ filename
        df = pd.read_csv(urldata, index_col=0)
        clf = joblib.load('model/' + model) 
        new_predict = prediction(clf, df)
        img_pred = chart_prediction(new_predict)
        #encoding y definitions
        
        resp = jsonify(new_predict, img_pred)
        resp.status_code = 201
        return resp
    else:
        resp = jsonify({'message' : 'Extensiones permitidas txt, pdf, png, jpg, jpeg, gif'})
        resp.status_code = 400
        return resp

def prediction(clf, df):
     #encoding data new
    le = LabelEncoder()
    ydef = pd.read_csv('data/y_definitions.csv', index_col=0)
    le.fit_transform(ydef)
    el = EncoderXY()
    ds_el= el.fit_transform(df)
    ds_el= ds_el.drop(['DEP_MUN_LUG'], axis=1)
    pred_df = clf.predict(ds_el) 
    #label prediction
    labels_pred = le.inverse_transform(pred_df)
    pred_df = pd.Series(labels_pred)
    pred_model = Counter(','.join(pred_df).replace('dest', 'count').split(',')).most_common(5)    
    return pred_model

def chart_prediction(X):
    df = pd.Series(X)
    
    #plot=df.apply(pd.value_counts).plot.pie(subplots=True)
    #plot = df.value_counts().plot(kind=pie, figsize=(5, 5))
    values = df.value_counts().values
    labels = df.value_counts().index
    
    plt.figure(figsize=(9,4))
    plt.pie(values, labels=labels, startangle=15, shadow = True, autopct=lambda p: '{:.2f}%({:.0f})'.format(p,(p/100)*values.sum()))
    plt.title('TOP Pronóstico de Destinos')

    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    pngfig = base64.b64encode(img.getvalue()).decode('ascii')
    img_name = datetime.now().strftime("%Y-%m-%d-%H:%M")
    with open("charts/prediction/predict"+img_name+ ".png", "wb") as fh:
        fh.write(base64.decodebytes(pngfig.encode()))
    #pngfig.save(os.path.join('', img_name))
    return pngfig #render_template('plot.html', plot_url=pngfig)


@app.route('/download', methods=['POST'])
def save():
    #obteniendo la data del form
    flag = False
    data = request.get_json(force=True)
    dataset = data['dataset'] 
    #label prediction
    algoritmo = data['alg']
    split = data['split']
    nsplit = int(split)/100
    
    # train split data
    urldata = 'data/'+ dataset
    df = pd.read_csv(urldata, index_col=0)
    EncodLabel = EncoderXY()
    X_le = EncodLabel.fit_transform(df)  # codificando todo el dataset
    X = X_le.drop(columns='DEP_MUN_LUG')  #MAtriz de variables X
    y = X_le['DEP_MUN_LUG'].copy()  # vector target y (predictive)
    
    # train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = nsplit, random_state = 42)
    
    #scaler
    scaler = StandardScaler()
    scaler.fit(X_train)   # standarizando todo el ds con Lencoder
    X_train_std = scaler.transform(X_train)
    X_test_std = scaler.transform(X_test)
    
    #shape
    xtrain = str(X_train.shape)
    ytrain = str(y_train.shape)
    xtest = str(X_test.shape)
    ytest = str(y_test.shape)

    #entrenoa
    if algoritmo =='PTRN':
        clf = Perceptron(penalty='l1', alpha=0.0001, fit_intercept=True, max_iter=1000, tol=0.001, shuffle=True, 
                 verbose=1, eta0=1.0, n_jobs=-1, random_state=1, early_stopping=False, validation_fraction=0.1, 
                 n_iter_no_change=5, class_weight=None, warm_start=False)
        clf.fit(X_train_std, y_train)
        model_saved = saving(algoritmo, clf)
        img= chart(X_test_std, clf, X_test, y_test, algoritmo)
        
    if algoritmo == 'RF':
        clf = RandomForestClassifier(n_estimators=200, criterion='gini', max_depth=None, min_samples_split=2, 
                            min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='sqrt', 
                            max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, 
                            bootstrap=True, oob_score=True, n_jobs=None, random_state=None, verbose=0, 
                            warm_start=True, class_weight=None) 
        clf.fit(X_train_std, y_train)
        model_saved = saving(algoritmo, clf)
        img= chart(X_test_std, clf, X_test, y_test, algoritmo)

    if algoritmo == 'KNN':
        clf = KNeighborsClassifier(n_neighbors=180, weights='distance', algorithm='auto', leaf_size=60, p=1, 
                           metric='minkowski', metric_params=None, n_jobs=None)
        clf.fit(X_train_std, y_train)
        model_saved = saving(algoritmo, clf)
        img= chart(X_test_std, clf, X_test, y_test, algoritmo)

    if algoritmo == 'SVC':
        clf = SVC(C=2.0, cache_size=200, class_weight=None, coef0=40.0, decision_function_shape='ovo', degree=3, gamma=0.020, 
                    kernel='poly',max_iter=-1, probability=True, random_state=12, shrinking=True,tol=0.00001, verbose=False)    
        clf.fit(X_train_std, y_train)
        model_saved = saving(algoritmo, clf)
        img= chart(X_test_std, clf, X_test, y_test, algoritmo)

    if algoritmo == 'MLP':
        clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(15,), random_state=1)
        clf.fit(X_train_std, y_train)
        model_saved = saving(algoritmo, clf)
        img= chart(X_test_std, clf, X_test, y_test, algoritmo)
    
    if algoritmo == 'SVR':
        clf = SVR(kernel='rbf', degree=3, gamma='scale', coef0=0.0, tol=0.001, C=1.0, epsilon=0.1, shrinking=True, cache_size=200, verbose=False, max_iter=-1)
        clf.fit(X_train_std, y_train)
        model_saved = saving(algoritmo, clf)
        img= chart(X_test_std, clf, X_test, y_test, algoritmo)
    
    resp = jsonify( xtrain, ytrain, xtest, ytest, dataset, algoritmo, nsplit, model_saved, img )
    resp.status_code = 200
    # return redirect('https://api-inguat.herokuapp.com/download/'+model_saved)
    return resp

def saving(algoritmo, clf):
    model =  algoritmo + '_model.joblib'
    joblib.dump(clf, 'tmp/' + model)
    return model

def chart(X, clf, X_test, y_test, algoritmo):
    y_pred = clf.predict(X)
    x_axis = X_test.EDAD
    plt.figure(figsize=(9,4))
    plt.scatter(x_axis, y_test, c = 'b', alpha = 0.9, marker = 'o')
    plt.scatter(x_axis, y_pred, c = 'r', alpha = 0.9, marker = 'o', label = 'Pronostico')
    plt.grid(color = '#D3D3D3', linestyle = 'solid')
    plt.legend(loc = 'middle right', fontsize=10, fancybox=True, shadow=True)
    
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    pngfig = base64.b64encode(img.getvalue()).decode('ascii')
    img_name = algoritmo + datetime.now().strftime("%Y-%m-%d-%H:%M")
    with open("charts/"+ img_name+ ".png", "wb") as fh:
        fh.write(base64.decodebytes(pngfig.encode()))
    #pngfig.save(os.path.join('', img_name))
    return pngfig #render_template('plot.html', plot_url=pngfig)

@app.route('/apidown/<path:filename>', methods=['GET'])
def download(filename):
    path = 'tmp/'+filename
    try:
        return send_file(path, as_attachment=True)
    # return send_file( path, as_attachment=True, mimetype='binary')
    except Exception as e:
	                    return str(e)


@app.route('/train', methods=['POST'])
def train():
    #obteniendo la data del form
    flag = False
    data = request.get_json(force=True)
    dataset = data['dataset']
    algoritmo = data['alg']
    split = data['split']
    nsplit = int(data['split'])/100
    
    # train split data
    urldata = 'data/'+ str(data['dataset'])
    df = pd.read_csv(urldata, index_col=0)
    EncodLabel = EncoderXY()
    X_le = EncodLabel.fit_transform(df)  # codificando todo el dataset
    X = X_le.drop(columns='DEP_MUN_LUG')  #MAtriz de variables X
    y = X_le['DEP_MUN_LUG'].copy()  # vector target y (predictive)
    
    # train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = nsplit, random_state = 42)
    
    #scaler
    scaler = StandardScaler()
    scaler.fit(X_train)   # standarizando todo el ds con Lencoder
    X_train_std = scaler.transform(X_train)
    X_test_std = scaler.transform(X_test)
    
    #shape
    xtrain = str(X_train.shape)
    ytrain = str(y_train.shape)
    xtest = str(X_test.shape)
    ytest = str(y_test.shape)

    #entrenoa
    if algoritmo =='PTRN':
        clf = Perceptron(penalty='l1', alpha=0.0001, fit_intercept=True, max_iter=1000, tol=0.001, shuffle=True, 
                 verbose=1, eta0=1.0, n_jobs=-1, random_state=1, early_stopping=False, validation_fraction=0.1, 
                 n_iter_no_change=5, class_weight=None, warm_start=False)
        clf.fit(X_train_std, y_train)
        y_pred = clf.predict(X_test_std)
        score = clf.score(X_test, y_test)
        acc= accuracy_score(y_pred, y_test)
        pred_data = decode(df, y_pred)
        
    if algoritmo == 'RF':
        clf = RandomForestClassifier(n_estimators=200, criterion='gini', max_depth=None, min_samples_split=2, 
                            min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='sqrt', 
                            max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, 
                            bootstrap=True, oob_score=True, n_jobs=None, random_state=None, verbose=0, 
                            warm_start=True, class_weight=None) 
        clf.fit(X_train_std, y_train)
        y_pred = clf.predict(X_test_std)
        score = clf.score(X_test, y_test)
        acc= accuracy_score(y_pred, y_test)
        pred_data = decode(df, y_pred)

    if algoritmo == 'KNN':
        clf = KNeighborsClassifier(n_neighbors=180, weights='distance', algorithm='auto', leaf_size=60, p=1, 
                           metric='minkowski', metric_params=None, n_jobs=None)
        clf.fit(X_train_std, y_train)
        y_pred = clf.predict(X_test_std)
        score = clf.score(X_test, y_test)
        acc= accuracy_score(y_pred, y_test)
        pred_data = decode(df, y_pred)
        #pred_list= pred_data.values.to_list()

    if algoritmo == 'SVC':
        clf = SVC(C=2.0, cache_size=200, class_weight=None, coef0=40.0, decision_function_shape='ovo', degree=3, gamma=0.020, 
                    kernel='poly',max_iter=-1, probability=True, random_state=12, shrinking=True,tol=0.00001, verbose=False)    
        clf.fit(X_train_std, y_train)
        y_pred = clf.predict(X_test_std)
        score = clf.score(X_test, y_test)
        acc= accuracy_score(y_pred, y_test)
        pred_data = decode(df, y_pred)

    if algoritmo == 'MLP':
        clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(15,), random_state=1)
        clf.fit(X_train_std, y_train)
        y_pred = clf.predict(X_test_std)
        score = clf.score(X_test, y_test)
        acc= accuracy_score(y_pred, y_test)
        pred_data = decode(df, y_pred)
    
    if algoritmo == 'SVR':
        clf = SVR(kernel='rbf', degree=3, gamma='auto_deprecated', coef0=0.0, tol=0.001, C=1.0, epsilon=0.1, shrinking=True, cache_size=200, verbose=False, max_iter=-1)
        clf.fit(X_train_std, y_train)
        y_pred = clf.predict(X_test_std)
        score = clf.score(X_test, y_test)
        acc= accuracy_score(y_pred, y_test)
        pred_data = decode(df, y_pred)
    

    resp = jsonify( xtrain, ytrain, xtest, ytest, dataset, algoritmo, nsplit, acc, pred_data )
    resp.status_code = 200
    return resp 

def decode(df, y_pred):
    le = LabelEncoder()
    le.fit_transform(df['DEP_MUN_LUG'])
    labels_pred = le.inverse_transform(y_pred)
    pred_df = pd.Series(labels_pred)
    #data=pred_df.value_counts()
    #data= data.nlargest(5)
    data= Counter(','.join(pred_df).replace('dest', 'count').split(',')).most_common(5)    
  
    return data
    
#********************Encoder CLASS---------------------------------------------------
class EncoderXY:
    def __init__(self, columns = None ):
        self.columns = columns    # lista de columnas a codificar

    def fit(self, X, y = None):
        return self

    def transform(self, X):
        '''
        transforma las columnas de X especificadas en self.columns usando Label_encoder().
        Si no se especifica transforma todas las columans de X
        '''
        salida = X.copy()

        if self.columns is not None:
            for col in self.columns:
                salida[col] = LabelEncoder().fit_transform(salida[col])
        else:
            for colname, col in salida.items():
                salida[colname] = LabelEncoder().fit_transform(col)

        return salida

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)

#*********************MODELS+++++++++++++++++++++++++++++
@app.route('/models', methods=['GET'])
def models():
    path = os.path.expanduser('tmp/')
    resp = jsonify(make_tree_model(path))
    return resp

def make_tree_model(path):
    tree = []
    name=os.path.basename(path)
    try: lst = os.listdir(path)
    except OSError:
        pass #ignore errors
    else:
        for name in lst:
            fn = os.path.join(path, name)
            tree.append(name)
    return tree

#---------------------------------CARGA de ARCHIVOS-----------------
@app.route('/carga_datos', methods= ['POST'])
def upload_file():
    if 'file' not in request.files:
        #flash('No file part')
        #return redirect('http://localhost:8080')
        resp = jsonify({'message' : 'No file part in the request'})
        resp.status_code = 400
        return resp
    file = request.files['file']
    if file.filename == '':
        resp = jsonify({'message' : 'No hay file seleccionada para cargar'})
        resp.status_code = 400
        return resp
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        resp = jsonify({'message' : 'File cargada exitosamente'})
        resp.status_code = 201
        df = pd.read_csv(UPLOAD_FOLDER+filename, index_col=0)
        #x = df 
        #return render_template("dataframe.html", name=filename, data=x)        
        # engine = create_engine('postgres://uwdzpuueaptugh:2b15054460b2c7b888fbe8fe213d685ba9384d4f013a58a4594b59b159c156e2@ec2-23-20-129-146.compute-1.amazonaws.com:5432/d75jsucusvnsh1')
        # for df in pd.read_csv(filename, names=columns,chunksize=1000):
        #     df.to_sql(
        #         'inguat_isnull', 
        #         engine,
        #         index=False,
        #         if_exists='append' # if the table already exists, append this data
        #     )
        #return redirect('http://localhost:8080/train')
        return resp
    else:
        resp = jsonify({'message' : 'Extensiones permitidas txt, pdf, png, jpg, jpeg, gif'})
        resp.status_code = 400
        return resp

###########---DESPLIEGUE de DATASETS
@app.route('/<filename>', methods=['GET'])
def load_dataset(filename):
    filename = request.view_args['filename']
    
    df = pd.read_csv(UPLOAD_FOLDER+filename, index_col=0)
    x = df.head() 
    # return render_template("dataframe.html", name=filename, data=x)
    # x = x.to_dict(orient='records')
    return x.to_json()

#*****************psql
@app.route('/upload', methods=['GET'])
def upload():
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        resp = jsonify({'message' : 'File cargada exitosamente'})
        resp.status_code = 201
    
    

#FIXED FOR FRONTEND--------------------------------------
@app.route('/datasets', methods=['GET'])
def datasets():
    path = os.path.expanduser(UPLOAD_FOLDER)
    # return render_template('file.html', tree=make_tree(path))
    tree = make_tree(path)
    return jsonify(tree)

def make_tree(path):
    tree = dict(path=os.path.basename(path), base=[])
    try: lst = os.listdir(path)
    except OSError:
        pass #ignore errors
    else:
        for name in lst:
            fn = os.path.join(path, name)
            if os.path.isdir(fn):
                tree['base'].append(make_tree(fn))
            else:
                with open(fn) as f:
                    contents = f.read()
                tree['base'].append(dict(name=name))
    return tree

@app.route('/dashboard', methods=['GET'])
def dashboard():
    chartpath= os.path.expanduser('/charts')
    resp = jsonify(make_tree_charts(chartpath))
    return resp

def make_tree_charts(path):
    tree = []
    name=os.path.basename(path)
    try: lst = os.listdir(path)
    except OSError:
        pass #ignore errors
    else:
        for name in lst:
            fn = os.path.join(path, name)
            tree.append(name)
    return tree

@app.route('/data_train', methods=['GET'])
def data_train():
    path = os.path.expanduser(UPLOAD_FOLDER)
    resp = jsonify(make_tree_train(path))
    return resp

def make_tree_train(path):
    tree = []
    name=os.path.basename(path)
    try: lst = os.listdir(path)
    except OSError:
        pass #ignore errors
    else:
        for name in lst:
            fn = os.path.join(path, name)
            tree.append(name)
    return tree

@app.route('/plot', methods=['GET'])
def plot_png():
    ds = pd.read_csv("data/Inguat_isnull.csv", index_col=0)
    x= ds['DEP_MUN_LUG']
    prob = x.value_counts(normalize=True)
    threshold = 0.007
    mask = prob > threshold
    tail_prob = prob.loc[~mask].sum()
    prob = prob.loc[mask]
    plt.xticks(rotation=15)
    plt.figure(figsize=(25,12))
    prob.plot(kind='barh')
    #
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    pngfig = base64.b64encode(img.getvalue()).decode('ascii')
    return render_template('plot.html', plot_url=pngfig)


def create_figure():
    fig = Figure()
    
    return fig
    
if __name__ == '__main__':
 app.run(debug=True)

CORS(app, expose_headers='Authorization')