import os

def read_all_documents(root):
    labels = []
    docs = []
    for r, dirs, files in os.walk(root):
        for file in files:
            with open(os.path.join(r, file), encoding="utf-8", mode="r") as f:
                docs.append(f.read())     
            labels.append(r.replace(root, ''))
    return dict([('docs', docs), ('labels', labels)])

data = read_all_documents('training')
documents = data['docs']
labels = data['labels']

import re
from collections import defaultdict

def tokens(doc):
    return (tok.lower() for tok in re.findall(r"\w+", doc))

def frequency(tokens):
    f = defaultdict(int)
    for token in tokens:
        f[token] += 1
    return f

def tokens_frequency(doc):
    return frequency(tokens(doc))

from sklearn.feature_extraction import DictVectorizer, FeatureHasher

vectorizer = DictVectorizer()
vectorizer.fit_transform(tokens_frequency(d) for d in documents)

vectorizer.get_feature_names()

#http://www.abc.es/economia/abci-bufetes-intentan-accionistas-bankia-vayan-juicio-201602190746_noticia.html
#hasher = FeatureHasher(n_features=2**8)
#X = hasher.transform(tokens_frequency(d) for d in documents)

hasher = FeatureHasher(n_features=2**8, input_type="string")
X = hasher.transform(tokens(d) for d in documents)

print(X.toarray())
print('---------------------------------------------------------')

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestNeighbors

prepositions =['a','ante','bajo','cabe','con','contra','de','desde','en','entre','hacia','hasta','para','por','según','sin','so','sobre','tras']
prep_alike = ['durante','mediante','excepto','salvo','incluso','más','menos']
adverbs = ['no','si','sí']
articles = ['el','la','los','las','un','una','unos','unas','este','esta','estos','estas','aquel','aquella','aquellos','aquellas']
aux_verbs = ['he','has','ha','hemos','habéis','han','había','habías','habíamos','habíais','habían']
tfid = TfidfVectorizer(stop_words=prepositions+prep_alike+adverbs+articles+aux_verbs)

X_train = tfid.fit_transform(documents)
y_train = labels

clf = KNeighborsClassifier(n_neighbors=10)
clf.fit(X_train, y_train)

test = read_all_documents('test')
X_test = tfid.transform(test['docs'])
y_test = test['labels']
pred = clf.predict(X_test)


print('accuracy score %0.3f' % clf.score(X_test, y_test))
print('---------------------------------------------------------')

#import eatiht.v2 as v2
import urllib.request
import requests
from bs4 import BeautifulSoup


def predict_category(url, classifier):    

    r = requests.get(url)
    html_content = r.text
    soup = BeautifulSoup(html_content, 'lxml')    
    article = soup.get_text()
    
    X_test = tfid.transform([article])
    return clf.predict(X_test)[0]

def show_predicted_categories(urls, classifier):
    for url in urls:
        print('Categorización de artículo: ' + predict_category(url, clf))

show_predicted_categories(
    [
        'https://trome.pe/mundial/brasil-vs-belgica-vivo-directo-tv-online-cuartos-final-mundial-rusia-2018-88466',
        'http://www.abc.es/economia/abci-bufetes-intentan-accionistas-bankia-vayan-juicio-201602190746_noticia.html',
        'http://www.elconfidencial.com/deportes/futbol/2016-02-19/torres-atletico-cope_1154857/',
        'http://archivo.elcomercio.pe/ciencias/investigaciones/vaticano-organiza-conferencia-sobre-agujeros-negros-noticia-1990705'],
    clf)
print('---------------------------------------------------------')