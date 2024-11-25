import json
import os
import sys
import pandas as pd
import numpy as np
import tensorflow as tf
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from keras import Sequential
from keras import layers
from keras import utils
from keras import callbacks
from keras_preprocessing.sequence import pad_sequences
from scipy.stats import mode
from sklearn.utils import class_weight

physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    print("Usando GPU para entrenamiento.")
else:
    print("No se encontró GPU. Usando CPU.")

#Cargar el dataset vectorizado
input_file_path = r'C:\Universidad\2024-2\Seminario de Investigacion II\Sentiment Analysis\RNN\vectorized_reviews.csv'
df = pd.read_csv(input_file_path)

# Convertir la columna de reseñas vectorizadas a listas
df['vector'] = df['vector'].apply(lambda x: list(map(int, x.split(','))))

print(df['sentimiento'].value_counts())

#tf.random.set_seed(42)
#np.random.seed(42)
#random.seed(42)

# Extraer las secuencias y etiquetas
X = np.array(df['vector'].tolist())
y = utils.to_categorical(df['sentimiento'] + 1)  # Convertir a one-hot (clases -1, 0, 1 -> índices 0, 1, 2) para binary class entropy

#Recalcular el diccionario word_index a partir de las secuencias
unique_words = set(word for sequence in X for word in sequence if word != 0)  # Excluir padding
vocab_size = len(unique_words) + 1  # +1 para incluir el índice 0 del padding




#division de conjuntos de prueba y entrenamiento 80 y 20 respectivamente
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#google colab
# display(X_train)

train_class_counts = np.argmax(y_train, axis=1)
test_class_counts = np.argmax(y_test, axis=1)
print("Distribución en el conjunto de entrenamiento:", np.bincount(train_class_counts))
print("Distribución en el conjunto de prueba:", np.bincount(test_class_counts))


# Parámetros
embedding_dim = 100          # Dimensión de los embeddings (debe coincidir con Word2Vec)
input_length = X_train.shape[1]  # Longitud máxima de las secuencias

mejor_precisión = 0.0  # Inicializar con una precisión baja
intento = 0            # Contador de intentos
historico = []  # Inicializa fuera del loop

while True:
    print(f"Intento: {intento + 1}")
    intento +=1
# Construcción del modelo RNN con LSTM
    model = Sequential([
        layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim, trainable=True),
        layers.LSTM(256, return_sequences=False),  # Capa LSTM con 256 unidades
        layers.Dropout(0.5),                       # Regularización
        layers.Dense(64, activation='relu'),
        layers.Dense(3, activation='softmax')      # 3 clases (positivo, neutral, negativo)
    ])
    early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
# Compilación del modelo
    model.compile(loss=tf.keras.losses.CategoricalCrossentropy(), optimizer=tf.keras.optimizers.Adam(), metrics=['accuracy'])

# Distribución original antes de sumar 1
    conteo_clases = {0: 1194, 1: 1788, 2: 2295}  # Basado en las etiquetas después de sumar 1
    max_samples = max(conteo_clases.values())

    class_weights = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(df['sentimiento']),
        y=df['sentimiento']
    )

    class_weights_dict = {i: weight for i, weight in enumerate(class_weights)}
    print(class_weights_dict)
#class_weights_dict2 = {0:1.4731993299832495, 1:1.08, 2:0.8}
#print(class_weights_dict2)

    checkpoint = callbacks.ModelCheckpoint(
        'mejor_modelo.keras', 
        monitor='val_accuracy', 
        save_best_only=True
    )
    # Entrenamiento
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, batch_size=16, class_weight=class_weights_dict, callbacks=[checkpoint, early_stopping])

    # Evaluación
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Pérdida: {loss:.4f}, Precisión: {accuracy:.4f}")
    
    # Almacenar métricas en la lista
    historico.append({
        'intento': intento,
        'accuracy': accuracy,
        'loss': loss
    })

# Guardar el modelo si mejora
    if accuracy > mejor_precisión:
        mejor_precisión = accuracy
        print(f"Nueva mejor precisión: {mejor_precisión:.4f}, guardando modelo...")
        model.save(f'mejor_modelo_intento_{intento}.keras')

# Mantener solo el mejor modelo actual
    for file in os.listdir('.'):
        if file.startswith('mejor_modelo_intento_') and not file.endswith(f'{intento}.keras'):
            os.remove(file)

    # Criterio de parada
    if mejor_precisión >= 0.8:  # Ajusta este valor según tu objetivo
        print("Se alcanzó la precisión deseada. Finalizando...")
        break
#graficar curvas

# Convertir historial a DataFrame
df_historico = pd.DataFrame(historico)

# Guardar historial en un archivo
df_historico.to_csv('historico_intentos.csv', index=False)

# Graficar métricas historicas
plt.figure(figsize=(10, 6))
plt.plot(df_historico['intento'], df_historico['loss'], label='Pérdida')
plt.plot(df_historico['intento'], df_historico['accuracy'], label='Precisión')
plt.xlabel('Intento')
plt.ylabel('Métrica')
plt.title('Evolución de la Pérdida y Precisión')
plt.legend()
plt.grid()
plt.show()



plt.plot(history.history['loss'], label='Pérdida Entrenamiento')
plt.plot(history.history['val_loss'], label='Pérdida Validación')
plt.legend()
plt.show()

plt.plot(history.history['accuracy'], label='Precisión Entrenamiento')
plt.plot(history.history['val_accuracy'], label='Precisión Validación')
plt.legend()
plt.show()

# Realizar predicciones
y_pred_prob = model.predict(X_test)
y_pred = np.argmax(y_pred_prob, axis=1)  # Convertir probabilidades a etiquetas (clase predicha)
y_true = np.argmax(y_test, axis=1)  # Convertir one-hot a etiquetas originales

# Mostrar ejemplos de predicciones junto con sus etiquetas reales
for i in range(10):  # Mostrar 10 ejemplos
    print(f"Reseña vectorizada: {X_test[i]}")
    print(f"Clase real: {y_true[i]}, Clase predicha: {y_pred[i]}")
    print("---")

y_pred_prob = model.predict(X_test)
print("Probabilidades de las primeras 10 predicciones:\n", y_pred_prob[:10])


print(classification_report(y_true, y_pred, target_names=["Negativo", "Neutral", "Positivo"]))

# Matriz de confusión
conf_matrix = confusion_matrix(y_true, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=["Negativo", "Neutral", "Positivo"], yticklabels=["Negativo", "Neutral", "Positivo"])
plt.xlabel('Clase Predicha')
plt.ylabel('Clase Real')
plt.show()

with open('pesos_clases.json', 'w') as f:
    json.dump(class_weights_dict, f)

with open('resultados.csv', 'a') as f:
    f.write(f"Intento {intento}, Precisión {accuracy:.4f}, Pérdida {loss:.4f}\n")
    f.flush()

sys.stdout = open('training_logs.txt', 'w')
sys.stderr = open('error_logs.txt', 'w')

output_file_path = r'C:\Universidad\2024-2\Seminario de Investigacion II\Sentiment Analysis\RNN\RNN_LSTM.keras'
model.save(output_file_path)
