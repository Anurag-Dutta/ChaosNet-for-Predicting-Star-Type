import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rc


dataset = pd.read_csv("data.csv")

O = dataset[dataset["Spectral Class"] == 'O']
B = dataset[dataset["Spectral Class"] == 'B']
A = dataset[dataset["Spectral Class"] == 'A']
F = dataset[dataset["Spectral Class"] == 'F']
G = dataset[dataset["Spectral Class"] == 'G']
K = dataset[dataset["Spectral Class"] == 'K']
M = dataset[dataset["Spectral Class"] == 'M']

plt.style.use(['default'])
# rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('font',**{'family':'serif','serif':['Calibri Light'], 'size':24})
rc('text', usetex=True)
plt.rcParams["figure.figsize"] = (16,16)
plt.scatter(np.log(O["Temperature (K)"]), np.log(O["Luminosity(L/Lo)"]),80, label = 'Hypergiant', color = 'white',edgecolors='black')
plt.scatter(np.log(B["Temperature (K)"]), np.log(B["Luminosity(L/Lo)"]),64, label = 'Supergiant', color = 'white',edgecolors='red')
plt.scatter(np.log(A["Temperature (K)"]), np.log(A["Luminosity(L/Lo)"]),48, label = 'Main Sequence', color = 'white',edgecolors='orange')
plt.scatter(np.log(F["Temperature (K)"]), np.log(F["Luminosity(L/Lo)"]),32, label = 'White Dwarf', color = 'cyan',edgecolors='blue')
plt.scatter(np.log(G["Temperature (K)"]), np.log(G["Luminosity(L/Lo)"]),24, color = 'cyan',edgecolors='blue')
plt.scatter(np.log(K["Temperature (K)"]), np.log(K["Luminosity(L/Lo)"]),16, label = 'Red Dwarf', color = 'red',edgecolors='red')
plt.scatter(np.log(M["Temperature (K)"]), np.log(M["Luminosity(L/Lo)"]),8, label = 'Brown Dwarf', color = 'white',edgecolors='brown')


#Sun
x = 9.7
a = 6.5
plt.scatter(x, a, 100, label = 'Sun', c='red',edgecolors='yellow')

plt.gca().invert_xaxis()
plt.ylabel("$log (Luminosity)$")
plt.xlabel("$log (Temperature)$")
plt.legend()
plt.savefig('HR.pdf', format="pdf")

