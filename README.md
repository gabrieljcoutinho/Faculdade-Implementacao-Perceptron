# Operação de Produto Escalar com NumPy 

Este repositório apresenta um exemplo prático de como realizar o cálculo de uma **combinação linear**, que é a operação base por trás de neurônios artificiais em Redes Neurais e modelos de Regressão.

##  Descrição do Projeto

O objetivo deste script é calcular a soma ponderada de um conjunto de entradas ($x$) multiplicadas por seus respectivos pesos ($w$), somadas a um termo de viés (*bias*).

### A Matemática por trás
A operação realizada é o produto escalar (dot product) entre dois vetores, definido pela fórmula:

$$z = \sum_{i=1}^{n} (w_i \cdot x_i) + b$$

No exemplo do código:
* $z = (0.1 \times 1) + (0.2 \times 2) + (0.3 \times 3) + 0$
* $z = 0.1 + 0.4 + 0.9 + 0$
* **$z = 1.4$**

---

##  O Código

```python
import numpy as np

# Entradas (Features)
x = np.array([1, 2, 3])

# Pesos (Weights)
w = np.array([0.1, 0.2, 0.3])

# Viés (Bias)
b = 0

# Cálculo: (w1*x1 + w2*x2 + w3*x3) + b
z = np.dot(w, x) + b

print(f"O resultado de z é: {z}")
