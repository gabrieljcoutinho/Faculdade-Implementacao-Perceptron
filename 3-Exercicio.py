#importando a biblioteca
import numpy as np

X = np.array([
[0,0],
[0,1],
[1,0],
[1,1]
])

y = np.array([0,0,0,1])

w,b = learn_weights(X,y,lr=0.1)

print("Pesos finais:", w)
print("Bias final:", b)

print("\nTestes:")
for i in range(len(X)):
    y_pred = perceptron(X[i],w,b,ActFunction="step")
    print(X[i], "->", y_pred, "| esperado:", y[i])

