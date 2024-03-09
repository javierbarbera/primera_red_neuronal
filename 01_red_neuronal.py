import tensorflow as tf
import numpy as np 


celsius = np.array([-40, -10, 0, 8, 15, 22, 38], dtype= float)
fahrenheit = np.array([-40, 14, 32, 46, 59, 72, 100], dtype= float)


red1 = tf.keras.layers.Dense(units = 3, input_shape = [1])
red2 = tf.keras.layers.Dense(units = 3)
salida = tf.keras.layers.Dense(units = 1)
modelo = tf.keras.Sequential([red1, red2, salida])


modelo.compile(
    optimizer= tf.keras.optimizers.Adam(0.1),
    loss = "mean_squared_error"
)

print("Comenzando entrenamiento...")
historial = modelo.fit(celsius, fahrenheit, epochs= 300, verbose = False)
print("Modelo entrenado")

print("Hagamos una predicción")
resultado = modelo.predict(np.array([100.0]))
print(resultado)

print("Variables internas del módelo")
#print(capa.get_weights())
print(red1.get_weights())
print(red2.get_weights())
print(salida.get_weights())