import pandas as pd
from corextopic import corextopic as ct
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import TextVectorization
from sklearn.model_selection import train_test_split

# Ruta del archivo
file_path = r'C:\Universidad\2024-2\Seminario II\Preprocesamiento_New\reseñas_falabellaUTF.csv'

# Cargar el dataset
df = pd.read_csv(file_path)

# Mostrar las primeras filas del dataset
print(df.head())

# Eliminar o rellenar los valores nulos en la columna 'texto_limpio'
df['texto_limpio'] = df['texto_limpio'].fillna('')

# Lista de palabras de parada en español
stop_words_spanish = ["de", "la", "que", "el", "en", "y", "a", "los", "del", "se", "las", "por", "un", "para", "con", "no", "una", "su", "al", "lo", "como", "más", "pero", "sus", "le", "ya", "o", "este", "sí", "porque", "esta", "entre", "cuando", "muy", "sin", "sobre", "también", "me", "hasta", "hay", "donde", "quien", "desde", "todo", "nos", "durante", "todos", "uno", "les", "ni", "contra", "otros", "ese", "eso", "ante", "ellos", "e", "esto", "mí", "antes", "algunos", "qué", "unos", "yo", "otro", "otras", "otra", "él", "tanto", "esa", "estos", "mucho", "quienes", "nada", "muchos", "cual", "poco", "ella", "estar", "estas", "algunas", "algo", "nosotros", "mi", "mis", "tú", "te", "ti", "tu", "tus", "ellas", "nosotras", "vosotros", "vosotras", "os", "mío", "mía", "míos", "mías", "tuyo", "tuya", "tuyos", "tuyas", "suyo", "suya", "suyos", "suyas", "nuestro", "nuestra", "nuestros", "nuestras", "vuestro", "vuestra", "vuestros", "vuestras", "esos", "esas", "estoy", "estás", "está", "estamos", "estáis", "están", "esté", "estés", "estemos", "estéis", "estén", "estaré", "estarás", "estará", "estaremos", "estaréis", "estarán", "estaría", "estarías", "estaríamos", "estaríais", "estarían", "estaba", "estabas", "estábamos", "estabais", "estaban", "estuve", "estuviste", "estuvo", "estuvimos", "estuvisteis", "estuvieron", "estuviera", "estuvieras", "estuviéramos", "estuvierais", "estuvieran", "estuviese", "estuvieses", "estuviésemos", "estuvieseis", "estuviesen", "estando", "estado", "estada", "estados", "estadas", "estad"]

# Preprocesamiento: Convertir el texto en una matriz de términos (bag of words)
vectorizer = CountVectorizer(stop_words=stop_words_spanish)
X = vectorizer.fit_transform(df['texto_limpio'])

# Definir el número de temas que deseas extraer
n_topics = 5  # Puedes ajustar este número

# Crear el modelo CorEx
corex_model = ct.Corex(n_hidden=n_topics)
corex_model.fit(X, words=vectorizer.get_feature_names_out(), anchors=[], anchor_strength=2)

# Obtener los temas
topics = corex_model.get_topics()

# Inspeccionar la estructura de los temas y desempaquetarlos correctamente
for i, topic in enumerate(topics):
    topic_words = [word for word, _ in topic]
    print(f'Topic {i}: {", ".join(topic_words)}')

# Obtener la categoría principal para cada documento
topic_labels = corex_model.transform(X)

# Añadir la categoría al dataframe original
df['category'] = np.argmax(topic_labels, axis=1)

# Guardar el dataset actualizado
output_path = r'C:\Universidad\2024-1\Seminario 1\Web Scraping\Preprocesamiento\textos_procesados_con_categorias.csv'
df.to_csv(output_path, index=False)

# Configurar la capa de vectorización de texto
max_features = 5000
sequence_length = 200

vectorize_layer = TextVectorization(
    max_tokens=max_features,
    output_mode='int',
    output_sequence_length=sequence_length
)

# Adaptar la capa de vectorización al texto
vectorize_layer.adapt(df['texto_limpio'])

# Vectorizar el texto
X = vectorize_layer(df['texto_limpio'])

# Crear las etiquetas (puede que necesites ajustar esto según tu formato de estrellas)
y = df['stars']  # Asumiendo que la columna de estrellas se llama 'stars'

# Dividir los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Construir el modelo RNN
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=max_features, output_dim=64, input_length=sequence_length),
    tf.keras.layers.LSTM(64, return_sequences=True),
    tf.keras.layers.LSTM(64),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Entrenar el modelo
model.fit(X_train, y_train, epochs=5, validation_data=(X_test, y_test), batch_size=32)
