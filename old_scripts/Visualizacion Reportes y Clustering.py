# -*- coding: utf-8 -*-

import os
import numpy as np
import matplotlib.pyplot as plt
from  sklearn.cluster import KMeans

#%%

# Load the data
ROOT_DATA = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data'))
cluster_dir = os.path.join(ROOT_DATA, 'Data Cluster')
data_path = os.path.join(cluster_dir, 'Datos_reportes_para_clusterizar_sin_reposo.csv')
data = np.loadtxt(data_path, skiprows=1, delimiter=",", dtype=float)

# Define variables
nsuj = 18
tfinal = 300
tiempo = np.linspace(0, 1200, 300)

# Dimensions for plotting
dimensiones = ['Pleasantness', 'Unpleasantness', 'Emotional_Intensity', 'Elementary_Imagery', 'Complex_Imagery',
               'Auditory', 'Interoception', 'Bliss', 'Anxiety', 'Entity', 'Selfhood', 'Disembodiment', 'Salience',
               'Temporality', 'General_Intensity']

# Number of subplots
num_plots = len(dimensiones)

# Create a figure and a set of subplots (3 rows, 5 columns for 15 dimensions)
fig, axs = plt.subplots(3, 5, figsize=(20, 12))

# Iterate over dimensions and plot each one in the corresponding subplot
for idx, ax in enumerate(axs.flat):
    alta = data[0:5400, idx]
    baja = data[5400:10800, idx]
    
    alta2 = np.reshape(alta, (nsuj, tfinal))
    baja2 = np.reshape(baja, (nsuj, tfinal))
    
    alta_promedio = alta2.mean(axis=0)
    alta_errores = alta2.std(axis=0) / np.sqrt(nsuj)
    
    baja_promedio = baja2.mean(axis=0)
    baja_errores = baja2.std(axis=0) / np.sqrt(nsuj)
    
    relleno_pos_alta = alta_promedio + alta_errores
    relleno_neg_alta = alta_promedio - alta_errores
    
    relleno_pos_baja = baja_promedio + baja_errores
    relleno_neg_baja = baja_promedio - baja_errores
    
    # Plot in the current subplot
    ax.plot(tiempo, alta_promedio, label='Dosis Alta', color="#9C27B0")
    ax.fill_between(tiempo, relleno_neg_alta, relleno_pos_alta, alpha=0.2, color="#9C27B0")
    
    ax.plot(tiempo, baja_promedio, label='Dosis Baja', color="#FFA726")
    ax.fill_between(tiempo, relleno_neg_baja, relleno_pos_baja, alpha=0.2, color="#FDD835")
    
    ax.set_title(dimensiones[idx])
    ax.set_xlabel('Tiempo (s)')
    ax.set_ylabel('Valor reportado')
    ax.set_ylim(0, 10)  # Set the y-axis limit from 0 to 10
    ax.legend()

# Adjust layout and show plot
plt.tight_layout()
plt.show()

#%% clustering 

# Load the data
data = np.loadtxt(data_path, skiprows=1, delimiter=",", dtype=float)
nsuj = 18
tfinal = 300

clus = KMeans(n_clusters=2).fit(data)
labels = clus.labels_

cluster1 = np.zeros(len(labels))
cluster1[np.where(labels==0)] = 1

cluster2 = np.zeros(len(labels))
cluster2[np.where(labels==1)] = 1

cluster1_alta = np.reshape(cluster1[0:5400], (nsuj,tfinal))
cluster2_alta = np.reshape(cluster2[0:5400], (nsuj,tfinal))

cluster1_baja= np.reshape(cluster1[5400:10800], (nsuj,tfinal))
cluster2_baja = np.reshape(cluster2[5400:10800], (nsuj,tfinal))

plt.close('all')

plt.figure()
plt.plot(tiempo, (cluster1_alta.mean(axis=0)), label = 'Estado 1', color = '#8BC34A')
plt.plot(tiempo, (cluster2_alta.mean(axis=0)), label = 'Estado 2', color = '#607D8B')
plt.ylabel('Probabilidad', fontsize=14)
plt.xlabel('Tiempo (s)', fontsize=14)
plt.yticks(fontsize=14)
plt.xticks(fontsize=14)
plt.legend(loc = 'center right')
plt.tight_layout()
plt.show()

plt.figure()
plt.plot(tiempo, (cluster1_baja.mean(axis=0)), label = 'Estado 1', color = '#8BC34A')
plt.plot(tiempo, (cluster2_baja.mean(axis=0)), label = 'Estado 2', color = '#607D8B')
plt.ylabel('Probabilidad', fontsize=14)
plt.xlabel('Tiempo (s)', fontsize=14)
plt.yticks(fontsize=14)
plt.xticks(fontsize=14)
plt.legend(loc = 'center right')
plt.tight_layout()
plt.show()

centroide1 = clus.cluster_centers_[0,:]/np.max(clus.cluster_centers_[0,:])
centroide2 = clus.cluster_centers_[1,:]/np.max(clus.cluster_centers_[1,:])

plt.figure()
plt.bar(dimensiones, centroide1, label = 'Centroide 1', color = '#8BC34A')
plt.ylabel('Dimensión normalizada', fontsize=18)
plt.yticks(fontsize=18)
plt.xticks(rotation = 45, ha='right', fontsize=18)
plt.legend(loc = 'upper right')
plt.tight_layout()
plt.show()

plt.figure()
plt.bar(dimensiones, centroide2, label = 'Centroide 2', color = '#607D8B')
plt.ylabel('Dimensión normalizada', fontsize=18)
plt.yticks(fontsize=18)
plt.xticks(rotation = 45, ha='right', fontsize=18)
plt.legend(loc = 'upper right')
plt.tight_layout()
plt.show()

#%% NO TIENE SENTIDO TENIENDO TANTASA DIMENSIONES: Ploteo de Clusters para chequear que funciona

#filter rows of original data
filtered_label1 = data[labels == 0]
filtered_label2 = data[labels == 1]


#Plotting the results
plt.scatter(filtered_label1[:,0] , filtered_label1[:,4] , color = 'red')
plt.scatter(filtered_label2[:,0] , filtered_label2[:,4] , color = 'black')
plt.show()

## Dos metodos para chequear si la cantidad de clusters es la mejor
#%% Silhoutte scores, debería mostrarme qué cantidad de clusters es la mejor

from sklearn.metrics import silhouette_score

# Define range of clusters to evaluate
cluster_range = range(2, 11)
silhouette_scores = []

# Run KMeans for each k and calculate the silhouette score
for k in cluster_range:
    kmeans = KMeans(n_clusters=k).fit(data)
    labels = kmeans.labels_
    # labels = kmeans.fit_predict(data)
    score = silhouette_score(data, labels)
    silhouette_scores.append(score)

plt.close('all')

# Plot Silhouette Score for each k
plt.figure(figsize=(8, 5))
plt.plot(cluster_range, silhouette_scores, marker='o', color='#3274a1')
plt.xlabel("Number of Clusters (k)", fontsize=14)
plt.ylabel("Silhouette Score", fontsize=14)
plt.xticks(cluster_range)
plt.grid(True)
plt.show()

#%% NO ME GUSTO PORQUE NUNCA TIENE UN ELBOW/ Elbow method: mismo fin que el anterior

# Define range of clusters to evaluate
cluster_range = range(1, 100)
wcss = []

# Run KMeans for each k and calculate the WCSS (inertia)
for k in cluster_range:
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(data)
    wcss.append(kmeans.inertia_)

# Plot WCSS (inertia) for each k
plt.figure(figsize=(8, 5))
plt.plot(cluster_range, wcss, marker='o', color='r')
plt.title("Elbow Method", fontsize=16)
plt.xlabel("Number of Clusters (k)", fontsize=14)
plt.ylabel("Inertia (WCSS)", fontsize=14)
plt.xticks(cluster_range)
plt.grid(True)
plt.show()

#%%










