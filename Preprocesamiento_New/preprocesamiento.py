from pysentimiento import create_analyzer
import transformers
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import spacy

# Configuración para pysentimiento
transformers.logging.set_verbosity(transformers.logging.ERROR)
analyzer = create_analyzer(task="sentiment", lang="es")

# Cargar los datos
df = pd.read_csv('reseñas_falabellaUTF.csv', encoding='ISO-8859-1')

#Borrar filas de la columna Texto con información no relevante
df = df[~df['Texto'].str.contains('¿Te fue útil este comentario?|Sin texto|Hola, lamentamos el inconveniente que has encontrado en nuestro producto. Para resolver todas tus dudas y brindarte una solución, te invitamos a ponerte en contacto con uno de nuestros expertos, comunícate al 800 SAMSUNG (726 7864) opción 2 o a través de nuestros medios digitales ingresando a https://www.samsung.com/mx/ y chatea con nosotros.|¡Felicidades! Nos alegra saber que estás feliz con tu compra.Saludos.', na=False
)]

textos = df['Texto']


# Descargar recursos de NLTK si no están descargados
nltk.download('punkt')
nltk.download('stopwords')

# Cargar SpaCy para lematización en español
nlp = spacy.load('es_core_news_sm')

# Pre-procesamiento
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

# Aplicar limpieza a cada texto
df['texto_limpio'] = textos.apply(limpiar_texto)

# Analizar el sentimiento de cada texto limpio
df['sentimiento'] = df['texto_limpio'].apply(lambda x: analyzer.predict(x).output)

# Guardar el resultado en un nuevo CSV
df.to_csv('textos_procesadossentimientomanual.csv', index=False)

