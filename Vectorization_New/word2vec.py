from gensim.models import Word2Vec
from nltk.tokenize import sent_tokenize, word_tokenize
import warnings
import pandas as pd
import numpy as np
from keras_preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import ast

input_file_path = r'C:\Universidad\2024-2\Seminario de Investigacion II\Sentiment Analysis\Vectorization_New\aug_text_vader_topics_utf8.csv'

df = pd.read_csv(input_file_path)

df['tokens'] = df['tokens'].apply(ast.literal_eval)

#Convertir la columna de sentimiento a valores numéricos
sentiment_mapping = {'POS': 1, 'NEG': -1, 'NEU': 0}
df['sentimiento'] = df['sentimiento'].map(sentiment_mapping)
df['sentimiento_vader'] = df['sentimiento_vader'].map(sentiment_mapping)

#modelo w2v
# Entrenamos con las reseñas tokenizadas (listas de palabras)
tokenized_reviews = df['tokens']
w2v_model = Word2Vec(sentences=tokenized_reviews, vector_size=100, window=5, min_count=1, workers=4)

# Crear un diccionario de palabra a índice
word_index = {word: idx + 1 for idx, word in enumerate(w2v_model.wv.index_to_key)}  # Indices empiezan desde 1
word_index['<PAD>'] = 0  # Índice especial para padding

#Vectorizar las reseñas como secuencias de índices
# Cada palabra se convierte en su índice correspondiente
reviews_sequences = [
    [word_index[word] for word in review if word in word_index] for review in tokenized_reviews
]

#Padding de las secuencias
# Uniformamos las longitudes de las secuencias para que sean compatibles con la RNN
max_sequence_length = max(len(seq) for seq in reviews_sequences)
padded_sequences = pad_sequences(reviews_sequences, maxlen=max_sequence_length, padding='post', value=0)

# Convertir las secuencias vectorizadas a cadenas para almacenarlas
df['vector'] = [','.join(map(str, seq)) for seq in padded_sequences]

# Guardar el dataset preprocesado
output_file_path = r'C:\Universidad\2024-2\Seminario de Investigacion II\Sentiment Analysis\Vectorization_New\vectorized_reviews.csv'
df.to_csv(output_file_path, index=False)

print(f"Dataset vectorizado guardado en: {output_file_path}")