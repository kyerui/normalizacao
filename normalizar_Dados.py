import pandas as pd
from pickle import dump

dados =pd.read_csv('ObesityDataSet_raw_and_data_sinthetic.csv', sep=',')
# print(dados.head(5))

dados_numericos = dados.drop(columns=['Gender','family_history_with_overweight', 'FAVC', 'CAEC','SMOKE','SCC','CALC','MTRANS','NObeyesdad' ])
dados_categoricos = dados[['Gender','family_history_with_overweight', 'FAVC', 'CAEC','SMOKE','SCC','CALC','MTRANS','NObeyesdad' ]]

dados_categoricos_normalizados = pd.get_dummies(data=dados_categoricos, dtype=int)
# print(dados_categoricos_normalizados)

colunas_categoricas = dados_categoricos_normalizados.columns
with open("colunas_categoricas.pkl", "wb") as f:
    dump(colunas_categoricas, f)

from sklearn import preprocessing
normalizador = preprocessing.MinMaxScaler()
modelo_normalizador = normalizador.fit(dados_numericos)

dados_numericos_normalizados = modelo_normalizador.fit_transform(dados_numericos)

dados_numericos_normalizados = pd.DataFrame(data = dados_numericos_normalizados, columns=['Age','Height','Weight','FCVC','NCP','CH2O','FAF','TUE'])

dados_normalizados_final = dados_numericos_normalizados.join(dados_categoricos_normalizados, how='left')

dados_normalizados_final_legiveis = modelo_normalizador.inverse_transform(dados_numericos_normalizados)

dados_normalizados_final_legiveis = pd.DataFrame(data= dados_normalizados_final_legiveis, columns=['Age','Height','Weight','FCVC','NCP','CH2O','FAF','TUE']).join(dados_categoricos_normalizados)
pd.set_option('display.max_columns', None)
print(dados_normalizados_final_legiveis)

dump(modelo_normalizador, open("normalizador.pkl", "wb"))


from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import math
from scipy.spatial.distance import cdist
import numpy as np

distortions = []
K = range(1, 101)

for i in K:
    obesity_kmeans_model = KMeans(n_clusters = i).fit(dados_normalizados_final_legiveis)
    distortions.append(sum(np.min(cdist(dados_normalizados_final_legiveis, obesity_kmeans_model.cluster_centers_, 'euclidean'), axis = 1)/dados_normalizados_final_legiveis.shape[0]))

print(distortions)

fig, ax = plt.subplots()
ax.plot(K, distortions)
ax.set(xlabel = 'n Clusters', ylabel = 'Distorção', title = 'Elbow pela distorção')
ax.grid()
fig.savefig('elbow_distorcao.png')
plt.show()

# Calcular o número ótimo de clusters
x0 = K[0]
y0 = distortions[0]
xn = K[len(K) - 1]
yn = distortions[len(distortions)-1]
# Iterar nos pontos gerados durante os treinamentos preliminares
distancias = []
for i in range(len(distortions)):
    x = K[i]
    y = distortions[i]
    numerador = abs((yn-y0)*x - (xn-x0)*y + xn*y0 - yn*x0)
    denominador = math.sqrt((yn-y0)**2 + (xn-x0)**2)
    distancias.append(numerador/denominador)

# Maior distância
n_clusters_otimo = K[distancias.index(np.max(distancias))]

obesity_kmeans_model = KMeans(n_clusters = n_clusters_otimo, random_state=42).fit(dados_normalizados_final)



dump(obesity_kmeans_model, open("obesity_cluster.pkl", "wb"))



# print(obesity_kmeans_model.cluster_centers_)






















