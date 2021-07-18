# Hierachical-Cluster

# Hierarchical clustering
from subprocess import check_output
import seaborn as sns  #as =  Zuweisung der Abkürzung der Bibliothek
import matplotlib.pyplot as plt

#Darstellungsstyle von sns in "white". -Darstellung in Farbe
sns.set(style="white", color_codes=True) 
 
#Führt zu statistischer Abbildung des Plots
%matplotlib inline 

#Import weiterer Bibliotheken
from pandas import Series, DataFrame
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.cluster import KMeans
from pylab import rcParams

# Festlegung Plot-Größe
rcParams['figure.figsize'] = 9, 8 

#Panda-Datenbank liest die Datei im angegebenen Dateipfad des eigenen PC aus
iris = pd.read_csv("C:/Users/User/Desktop/Felix/UNI/Masterstudium/Machine Learning/Übungsdaten/iris.csv")

#Erstellung Liste ,,iris_SP“ aus dem Dataset ,,iris“ mit den Spaltenname ,,sepal_length“, ,,sepal_width“ usw.
iris_SP = iris[['sepal_length','sepal_width','petal_length','petal_width']]

#Aufrufen von ,,iris_SP“ Head: Aufruf des Listenkopfes
iris_SP.head()

#Describe als Funktion
#Auswertung der ,,iris_SP"-Liste anhand deskriptiver Statistik (Minimum, Maximum, Unteres Quartil,…)
iris_SP.describe()

# model3 wird ein K-Means mit 2 Clustern zugewiesen
model3=KMeans(n_clusters=2)

#Iris_SP wird mit fit()-Funktion an model3 mit 2 Clustern angepasst
model3.fit(iris_SP) 

#Zuweisung der clussassign-Variable: Vorhersagung von Werten aus iris_SP
clusassign=model3.predict(iris_SP)

#Importieren PCA 
from sklearn.decomposition import PCA 

#Zuweisung der Funktion PCA(2) -> Zurückgeben zweier kanonischer Variablen 
pca_2 = PCA(2)

#Dataset aus pca_2 wird zu Trainings-Dataset iris_SP angepasst
plot_columns = pca_2.fit_transform(iris_SP) 
plt.scatter(x=plot_columns[:,0], y=plot_columns[:,1], c=model3.labels_,)


#Achsenzuweisung (x und y bilden die beiden Cluster mit den Punkten c aus model13.labels ab)
plt.xlabel('Canonical variable 1') #Achsenbeschriftung x-Achse: "Canonical variable 1"
plt.ylabel('Canonical variable 2') #Achsenbeschriftung y-Achse: "Canonical variable 2"
plt.title('Scatterplot of Canonical Variables for 2 Clusters') #Titel: "Scatterplot of Canonical Variables for 2 Clusters"
plt.show() #Anzeigen des Diagramms

# Importieren Dendrogramm, Linkage
from scipy.cluster.hierarchy import dendrogram, linkage

# Z: Zuweisung und Generierung der Linkage Matrix mit Dataset iris-SP und "ward" zur Minimierung der Varianz zwischen den Clustern 
Z = linkage(iris_SP, 'ward')

#max_d: Zuweisung maximaler Distanz im Cluster
max_d = 7.08 

#Bestimmung Anzeigegröße mithilfe von figsize auf (25,10)
plt.figure(figsize=(25, 10))
plt.title('Iris Hierarchical Clustering Dendrogram') #Titelbeschriftung: "Iris Hierarchical Clustering Dendrogram"
plt.xlabel('Species') #Achsenbeschriftung x-Achse: "Species"
plt.ylabel('distance') #Achsenbeschriftung y-Achse: "distance"


#Durchführung Funktion Dendrogramms
dendrogram(
    Z, #Z = Linkage Matrix (Verbindung zum Datensatz Iris)
    truncate_mode='lastp', # Zuweisen letzter p der zusammengeführten Cluster mit truncate_mode="lastp"
    p=10, # p: Zuweisung Clusteranzahl = 10
    leaf_rotation=90., # Leaf Rotation: Rotation der X-Achsen-Beschriftung -> 90°
    leaf_font_size=15., # Leaf Font Size: Schriftgröße der X-Achsen-Beschriftung -> 8

)

#Axhline erstellen: Horizontale Linie bei y=7.08 (max_d) mit Darstellungstyp "k"
plt.axhline(y=max_d, c='k') 

#Anzeigen des Dendrogramms
plt.show()

# calculate full dendrogram for 50
from scipy.cluster.hierarchy import dendrogram, linkage
# generate the linkage matrix
Z = linkage(iris_SP, 'ward')
# set cut-off to 50
max_d = 7.08 # max_d as in max_distance
plt.figure(figsize=(25, 10))
plt.title('Iris Hierarchical Clustering Dendrogram')
plt.xlabel('Species')
plt.ylabel('distance')
dendrogram(
    Z,
    truncate_mode='lastp', # show only the last p merged clusters
    p=50, # Try changing values of p
    leaf_rotation=90., # rotates the x axis labels
    leaf_font_size=8., # font size for the x axis labels
)
plt.axhline(y=max_d, c='k')
plt.show()
