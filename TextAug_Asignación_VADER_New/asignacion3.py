import pandas as pd

# Definir la ruta del archivo de entrada y salida
input_file_path = r'C:\Universidad\2024-2\Seminario de Investigacion II\Sentiment Analysis\TextAug_Asignación_VADER_New\aug_text_vade_utf8.csv'
output_file_path = r'C:\Universidad\2024-2\Seminario de Investigacion II\Sentiment Analysis\TextAug_Asignación_VADER_New\aug_text_vader_topics_utf8.csv'

# Cargar el dataset con modelado de temas
data = pd.read_csv(input_file_path)

# Asegurarse de que no haya valores nulos en 'texto_limpio'
data['texto_limpio'] = data['texto_limpio'].fillna('')

# Mapeo de tópicos a categorías
topic_mapping = {
    0: 'Rendimiento',
    1: 'Bateria',
    2: 'Camara'
}

# Asignar la categoría de tópico basada en dominant_topic
data['category'] = data['dominant_topic'].map(topic_mapping)

# borrar reseñas con texto_limpio o tokens vacío
data = data[(data['texto_limpio'] != '') & (data['tokens'] != '[]')]

# Guardar el dataset con las nuevas columnas
data.to_csv(output_file_path, index=False)

print("Dataset guardado con éxito en la ruta especificada.")
