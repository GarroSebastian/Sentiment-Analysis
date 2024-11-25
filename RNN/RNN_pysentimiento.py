import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from keras import Sequential
from keras import layers
from keras import utils
from keras_preprocessing.sequence import pad_sequences
from scipy.stats import mode
from sklearn.utils import class_weight

#Cargar el dataset vectorizado
input_file_path = r'C:\Universidad\2024-2\Seminario de Investigacion II\Sentiment Analysis\RNN\vectorized_reviews.csv'
df = pd.read_csv(input_file_path)

# Convertir la columna de reseñas vectorizadas a listas
df['vector'] = df['vector'].apply(lambda x: list(map(int, x.split(','))))

print(df['sentimiento'].value_counts())

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


# Parámetros
embedding_dim = 100          # Dimensión de los embeddings (debe coincidir con Word2Vec)
input_length = X_train.shape[1]  # Longitud máxima de las secuencias

# Construcción del modelo RNN con LSTM
model = Sequential([
    layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim, trainable=False),
    layers.LSTM(256, return_sequences=False),  # Capa LSTM con 256 unidades
    layers.Dropout(0.5),                       # Regularización
    layers.Dense(64, activation='relu'),
    layers.Dense(3, activation='softmax')      # 3 clases (positivo, neutral, negativo)
])

# Compilación del modelo
model.compile(loss=tf.keras.losses.CategoricalCrossentropy(), optimizer=tf.keras.optimizers.Adam(), metrics=['accuracy'])

class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(df['sentimiento']),
    y=df['sentimiento']
)

class_weights_dict = {i: weight for i, weight in enumerate(class_weights)}


# Entrenamiento
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, batch_size=16, class_weight=class_weights_dict)

# Evaluación
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Pérdida: {loss:.4f}, Precisión: {accuracy:.4f}")

#graficar curvas

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

print(classification_report(y_true, y_pred, target_names=["Negativo", "Neutral", "Positivo"]))

# Matriz de confusión
conf_matrix = confusion_matrix(y_true, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=["Negativo", "Neutral", "Positivo"], yticklabels=["Negativo", "Neutral", "Positivo"])
plt.xlabel('Clase Predicha')
plt.ylabel('Clase Real')
plt.show()

# Curvas ROC para cada clase
fpr = {}
tpr = {}
roc_auc = {}
n_classes = 3
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve((y_test == i).astype(int), y_pred_prob[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Gráfico de todas las curvas ROC
plt.figure()
for i in range(n_classes):
    plt.plot(fpr[i], tpr[i], lw=2, label=f'Clase {i} (área = {roc_auc[i]:0.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Tasa de Falsos Positivos')
plt.ylabel('Tasa de Verdaderos Positivos')
plt.title('Curvas ROC para cada clase')
plt.legend(loc="lower right")
plt.show()

output_file_path = r'C:\Universidad\2024-2\Seminario de Investigacion II\Sentiment Analysis\RNN\RNN_LSTM.keras'
model.save(output_file_path)
