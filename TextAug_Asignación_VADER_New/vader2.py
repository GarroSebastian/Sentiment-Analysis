import pandas as pd
import matplotlib.pyplot as plt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

file_path = r'C:\Universidad\2024-2\Seminario de Investigacion II\Sentiment Analysis\TextAug_Asignación_VADER_New\aug_text_utf.csv'
df = pd.read_csv(file_path, encoding='ISO-8859-1')

# Inicializar el analizador VADER
analyzer = SentimentIntensityAnalyzer()

# Función para obtener el sentimiento en español usando VADER
def obtener_sentimiento_vader(texto):
    # Obtener los puntajes de sentimiento
    puntajes = analyzer.polarity_scores(texto)
    # Clasificar según el puntaje compuesto
    if puntajes['compound'] >= 0.05:
        return 'POS'  # Positivo
    elif puntajes['compound'] <= -0.05:
        return 'NEG'  # Negativo
    else:
        return 'NEU'  # Neutral

# Aplicar el análisis de sentimiento a la columna 'Texto' o 'texto_limpio'
df['sentimiento_vader'] = df['Texto'].apply(obtener_sentimiento_vader)

conteo_sentimientos = df.groupby(['dominant_topic', 'sentimiento_vader']).size().unstack(fill_value=0)

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


df.to_csv('aug_text_vader_utf.csv', encoding='utf-8')
