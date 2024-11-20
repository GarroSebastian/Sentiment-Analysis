import pandas as pd
from pysentimiento import create_analyzer
from collections import Counter
import transformers
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import spacy
import matplotlib.pyplot as plt

# Configuración para pysentimiento
transformers.logging.set_verbosity(transformers.logging.ERROR)
analyzer = create_analyzer(task="sentiment", lang="es")


# Descargar recursos de NLTK si no están descargados
nltk.download('punkt')
nltk.download('stopwords')

# Cargar SpaCy para lematización en español
nlp = spacy.load('es_core_news_sm')

# Función de preprocesamiento
def limpiar_texto(texto):
    # Minúsculas
    texto = texto.lower()
    # Eliminar emojis y otros caracteres no alfabéticos
    texto = re.sub(r'[^\w\s]', '', texto)
    texto = re.sub(r'[0-9]+', '', texto)
    texto = re.sub(r'\s+', ' ', texto).strip()
    # Tokenización
    palabras = word_tokenize(texto, language='spanish')
    # Eliminar palabras cortas y stop words
    palabras = [palabra for palabra in palabras if len(palabra) > 2]
    stop_words = set(stopwords.words('spanish'))
    palabras = [palabra for palabra in palabras if palabra not in stop_words]
    # Lematización
    texto_lematizado = " ".join([token.lemma_ for token in nlp(" ".join(palabras))])
    return texto_lematizado

file_path = r'C:\Universidad\2024-2\Seminario de Investigacion II\Sentiment Analysis\TextAug_Asignación_VADER_New\textos_procesados_con_temas_utf.csv'
df = pd.read_csv(file_path, encoding= 'utf-8')


# Filtrar solo las filas con 'sentimiento' igual a 'POS'
df_pos = df[df['sentimiento'] == 'POS'].copy()

# Combinar texto de las columnas 'Texto', 'texto_limpio' y 'tokens' en un solo string
all_text = " ".join(df_pos['Texto'].astype(str)) + " " + " ".join(df_pos['texto_limpio'].astype(str)) + " " + " ".join(df_pos['tokens'].astype(str))

# Eliminar puntuación y convertir a minúsculas
all_text_cleaned = re.sub(r"[^\w\s]", "", all_text.lower())

# Tokenizar palabras y contar frecuencia
word_counts = Counter(all_text_cleaned.split())

# Mostrar las palabras más comunes
most_common_words = word_counts.most_common(100)  # Cambia el número para más o menos palabras
print(most_common_words)

# Palabras más comunes detectadas y sus reemplazos
replacements = {
    "excelente": "decepcionante",
    "buen": "malo",
    "Buen": "Mal",
    "buena": "mala",
    "correcto": "incorrecto",
    "nítido": "borroso",
    "colorido": "feo",
    "alegra": "deprime",
    "buenas": "malas",
    "Bello": "horrible",
    "innovador": "obsoleto",
    "moderno": "obsoleto",
    "buenaY": "mala",
    "bien": "mal",
    "bueno": "malo",
    "bonito": "horrible",
    "genial": "terrible",
    "mejor": "peor",
    "súper": "pésimo",
    "cumplir": "incumplir",
    "super": "pésimo",
    "satisfecho": "insatisfecho",
    "perfecto": "defectuoso",
    "recomendado": "no recomendado",
    "fácil": "difícil",
    "rápido": "lento",
    "duradero": "frágil",
    "útil": "inservible",
    "nuevo": "obsoleto",
    "duro": "frágil",
    "rapido": "lento",
    "lindo": "horrible",
    "encanto": "decepcionó",
    "siempre": "nunca",
    "gran": " ",
    "liviano": "pesado",
    "original": "trivial",
    "mucho": "poco",
    "mismas": "distintas",
    "mismo": "distinto",
    "hermoso": "horrible",
    "igualito": "distinto",
    "igual": "distinto",
    "encantar": "odiar",
    "feliz": "deprimido",
    "exelente": "pésimo",
    "cómodo": "incómodo"
}

# Aplicar reemplazos dinámicamente
df_pos['Texto'] = df_pos['Texto'].apply(lambda x: " ".join([replacements.get(word, word) for word in x.split()]))

# Aplicar limpieza al texto y generar tokens para las filas generadas
df_pos['texto_limpio'] = df_pos['Texto'].apply(limpiar_texto)
df_pos['tokens'] = df_pos['texto_limpio'].apply(lambda x: x.split())

# Analizar el sentimiento usando pysentimiento (aunque ya sabemos que son negativas, asegura consistencia)
df_pos['sentimiento'] = df_pos['texto_limpio'].apply(lambda x: analyzer.predict(x).output)

# Combinar el DataFrame original con el DataFrame de las nuevas filas
df_final = pd.concat([df, df_pos], ignore_index=True)

# Exportar el DataFrame combinado a un nuevo archivo CSV con codificación UTF-8
df_final.to_csv("augText_utf.csv", index=False, encoding='utf-8')

conteo_sentimientos = df_final.groupby(['dominant_topic', 'sentimiento']).size().unstack(fill_value=0)

conteo_sentimientos.plot(kind='bar', stacked=True, figsize=(12, 6), colormap='viridis')

# Personalizar el gráfico
plt.title('Cantidad de reseñas por tópico y sentimiento')
plt.xlabel('Tópico')
plt.ylabel('Cantidad de reseñas')
plt.legend(title='Sentimiento')
plt.xticks(rotation=0)
plt.tight_layout()

# Mostrar el gráfico
plt.show()
