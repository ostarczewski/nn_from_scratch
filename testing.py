from layer import Dense
from activation import ReLU
import numpy as np

x1 = np.array([1.5, 2.4, 3])

network = [Dense(3,10), ReLU(), Dense(10,1)]

print(f"""
Base weights:
{network[0].weights}
{network[2].weights}
""")

output = x1
for layer in network:
    output = layer.forward(output)

print(output)
