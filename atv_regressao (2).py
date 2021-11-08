import random
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

## Salvar o data frame em .csv
#df.to_csv("reg.csv")
np.random.seed(44)
random.seed(44)

#Gerando os dados das dristrbuições Normal, Uniforme, Exponencial e Beta.
n = 1000000
dset = {}
for i in range(1,40):           # Normal(0, aleatorio entre 0 e 1)
    chave = 'x' + str(i)
    valor = np.random.normal(0,random.random(),n)
    dset[chave] = valor

for i in range(40,80):         # Uniforme
    chave = 'x' + str(i)
    valor = np.random.rand(n)
    dset[chave] = valor

for i in range(80,120):         # Exponencial(Valores aleatorios entre 1 e 10)
    chave = 'x' + str(i)
    valor = np.random.exponential(random.randrange(1, 11),n)
    dset[chave] = valor    

for i in range(120,160):         # Beta(Valores aleatorios entre 1 e 5, Valores aleatorios entre 1 e 5)
    chave = 'x' + str(i)
    valor = np.random.beta(random.randrange(1, 5),random.randrange(1, 5),n)
    dset[chave] = valor

for i in range(160,200):         #Gamma(Valores aleatorios entre 1 e 5, Valores aleatorios entre 1 e 5)
    chave = 'x' + str(i)
    valor = np.random.gamma(random.randrange(1, 5),random.randrange(1, 5),n)
    dset[chave] = valor

df = pd.DataFrame(dset)
print(df)

X = df.iloc[:,list(range(199))].values
type(X)
print(X)

betas = []
for j in range(1,200):
    b = np.random.normal(0,2)
    betas.append(b)

soma = 0
for l in range(0,199):
    soma = soma + betas[l]*X[:,l]

y = 10+soma+np.random.normal(0,1,n)
print(y)

reg = LinearRegression()
reg.fit(X,y)
print(reg.coef_)
print(reg.intercept_)

sns.heatmap(df.corr(),square = True, cmap = "YlGnBu")
plt.show()

pd.plotting.scatter_matrix(df.iloc[:, 0:10], grid = False, alpha = 0.2, figsize = (12,12), diagonal = "kde" )         # 
plt.show()


pd.plotting.scatter_matrix(df.iloc[:, 20:40])       # # Mostrar x20 ate x40
plt.show()

pd.plotting.scatter_matrix(df.iloc[:, 40:60])       # # Mostrar x40 ate x60
plt.show()

pd.plotting.scatter_matrix(df.iloc[:, 60:80])       # # Mostrar x60 ate x80
plt.show()

pd.plotting.scatter_matrix(df.iloc[:, 80:100])       # # Mostrar x80 ate x100
plt.show()

pd.plotting.scatter_matrix(df.iloc[:, 100:120])       # # Mostrar x100 ate x120
plt.show()

pd.plotting.scatter_matrix(df.iloc[:, 120:140])       # # Mostrar x120 ate x140
plt.show()

pd.plotting.scatter_matrix(df.iloc[:, 140:160])       # # Mostrar x140 ate x160
plt.show()

pd.plotting.scatter_matrix(df.iloc[:, 160:180])       # # Mostrar x160 ate x180
plt.show()

pd.plotting.scatter_matrix(df.iloc[:, 180:200])       # # Mostrar x180 ate x200
plt.show()

