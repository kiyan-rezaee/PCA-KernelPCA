import pandas as pd
import hazm
from hazm import word_tokenize

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