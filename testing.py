from layer import Dense
from activation import ReLU
import numpy as np

            #  x1   x2   x3
x = np.array([[1.5, 2.4, 3.0],
              [1.0, 2.0, 3.6],
              [1.0, 2.0, 3.7],
              [1.2, 2.1, 2.6],
              [1.4, 2.0, 3.8]])
# size: batch x liczba cech

network = [Dense(3,10), ReLU(), Dense(10,1)]

print(f"""
Base weights:
{network[0].weights}
{network[2].weights}
""")

y_pred = x
for layer in network:
    y_pred = layer.forward(y_pred)

print(y_pred)
