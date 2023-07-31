import sent2vec
from nltk import word_tokenize
from nltk.corpus import stopwords
from string import punctuation
from scipy.spatial import distance
import sqlite3 as sq
from sqlite3 import Error
import json


stop_words = set(stopwords.words('english'))
database_path = "/scratch/nimoosterlaar/database.sqlite"
model_path = '/scratch/nimoosterlaar/create_sentence_embeddings/sent2vec/examples/not_works.bin'


def create_connection(db_file):
    try:
        return sq.connect(db_file)
    except Error as e:
        print(e)
    
    return None

def run(parameter, select = True, batch = False):
    conn = create_connection(database_path)
    result = None
    try:
        cur = conn.cursor()
        if batch:
            for command in parameter():
                # print("Executing: " + command + "\n")
                cur.execute(command)   
        else:    
            cur.execute(parameter)
        if select:
            result = cur.fetchall()
        conn.commit()
        cur.close()
    except Error as error:
        print(error)
    finally:
        if conn is not None:
            conn.close()
    return result


def preprocess_sentence(text):
    text = text.replace('/', ' / ')
    text = text.replace('.-', ' .- ')
    text = text.replace('.', ' . ')
    text = text.replace('\'', ' \' ')
    text = text.lower()

    tokens = [token for token in word_tokenize(text) if token not in punctuation and token not in stop_words]

    return ' '.join(tokens)


def make_sentence_embedding(output_file = '/scratch/nimoosterlaar/create_sentence_embeddings/sentence_embedding.json'):
    embeddings = {}
    
    model = sent2vec.Sent2vecModel()
    try:
        model.load_model(model_path)
    except Exception as e:
        print(e)
    print('model successfully loaded')
    
    go_terms = retrieve_GO_term()
    for go_term in go_terms:
        # print(go_term)
        sentence = preprocess_sentence(go_term[1])
        embeddings[go_term[0]] = model.embed_sentence(sentence)[0].tolist()
    
    # print(embeddings)
    
    with open(output_file, 'w') as f:
        json.dump(embeddings, f)
    
    
    

def retrieve_GO_term():
    query = "SELECT id, term FROM go_term;"
    result = run(query)
    return result

# string_1 = preprocess_sentence('Breast cancers with HER2 amplification have a higher risk of CNS metastasis and poorer prognosis.')
# string_2 = preprocess_sentence('Breast cancers with HER2 amplification are more aggressive, have a higher risk of CNS metastasis, and poorer prognosis.')
# string_3 = preprocess_sentence('Furthermore, increased CREB expression in breast tumors is associated with poor prognosis, shorter survival and higher risk of metastasis.')
# sentence_vector1 = model.embed_sentence(string_1)[0]
# sentence_vector2 = model.embed_sentence(string_2)[0]
# sentence_vector3 = model.embed_sentence(string_3)[0]

# # print(string_1)
# # print(sentence_vector1)
# # print(sentence_vector1.shape)

# cosine_sim = 1 - distance.cosine(sentence_vector1, sentence_vector2)
# print('cosine similarity:', cosine_sim)

# cosine_sim = 1 - distance.cosine(sentence_vector1, sentence_vector3)
# print('cosine similarity:', cosine_sim)


if __name__ == '__main__':
    make_sentence_embedding()