import pandas as pd
import hazm
from hazm import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv('per.csv')
with open('stopwords.txt') as stopwords_file:
    stopwords = stopwords_file.readlines()
stopwords = [line.replace('\n', '') for line in stopwords]

stemmer = hazm.Stemmer()
lem = hazm.Lemmatizer()

dataset = pd.DataFrame(columns=('title_body', 'category'))
for index, row in data.iterrows():
    title_body = row['Title'] + ' ' + row['Body']
    title_body_tokenized = word_tokenize(title_body)
    title_body_tokenized_filtered = [w for w in title_body_tokenized if not w in stopwords]
    title_body_tokenized_filtered_stemmed = [stemmer.stem(w) for w in title_body_tokenized_filtered]
    title_body_tokenized_filtered_lem = [lem.lemmatize(w).replace('#', ' ') for w in title_body_tokenized_filtered]
    dataset.loc[index] = {
        'title_body': ' '.join(title_body_tokenized_filtered_lem) + ' ' + ' '.join(title_body_tokenized_filtered_stemmed),
        'category': row['Category2'].replace('\n', '')
    }

vectorizer = TfidfVectorizer()
vectorizer.fit(dataset['title_body'])
X = vectorizer.transform(dataset['title_body'])

le = LabelEncoder()
y = le.fit_transform(dataset['category'])

from sklearn.decomposition import PCA, KernelPCA
kpca = KernelPCA(n_components=100)
kpca.fit(X)
X = kpca.transform(X)

# Evaluation 

from sklearn.model_selection import train_test_split
from sklearn import svm

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=321)
svmc = svm.SVC()
svmc.fit(X_train, y_train)
svmc.score(X_test, y_test)

from sklearn.metrics import classification_report, confusion_matrix
y_pred = svmc.predict(X_test)

print(classification_report(y_test, y_pred))
print(classification_report(y_test, y_pred))