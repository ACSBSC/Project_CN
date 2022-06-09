#################### Imports ####################
import numpy as np
import os
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt

'''
Lokta Volterra 

aN -bNP = cNP - dP (estado estacionario)

N -> presa
P -> depredador
a -> reproducción presa
d -> reproducción depredador
b -> tasa de cambio debida a la interacción
c -> tasa de cambio debida a la interacción

Lokta Volterra del paper

rH^k - qCH = gqCH - mC (estado estacionario)

H -> presa
C -> depredador
r -> reproducción presa
g -> reproducción depredador
m -> tasa de mortalidad depredador
q -> fuerza de interacción trófica (no 100 porciento necesaria)
k -> exponente de escala de la comunidad de presas (tampoco necesaria)
'''



#################### Settings ####################
CSV_FILEPATH = os.path.join("FW_005", "FW_005.csv")
CSV_FILEPATH_2 = os.path.join("FW_005", "FW_005-species.csv")
SELF_LIMITATION = 0.1
MUTUALISM = 0.1
COMPETITION = 0.001

REPRODUCTION_RATE_PRAY = 1.5
REPRODUCTION_RATE_PREDATOR = 1
PREDATOR_MORTALITY = 1.2

#################### Functions ####################
def create_iteraction_matrix(filepath=CSV_FILEPATH, self_limitation=SELF_LIMITATION,
                             mutualism=MUTUALISM, competition=COMPETITION):
    # Read mutualism matrix
    mutualism_mat = np.genfromtxt(filepath, delimiter=',', dtype=float)
    num_individuals = mutualism_mat.shape[0]

    # Remove canibalism
    np.fill_diagonal(mutualism_mat, 0)

    # Make weights to sum 1
    col_weights = mutualism_mat.sum(0)
    np.divide(mutualism_mat, col_weights, where=col_weights != 0, out=mutualism_mat)

    # Compute competition matrix
    competition_mat = np.zeros_like(mutualism_mat)
    for i in range(num_individuals):
        for j in range(num_individuals):
            if i != j:
                competition_mat[i, j] = np.sum(mutualism_mat[:, j], where=mutualism_mat[:, i] > 0)

    # Create interactions matrix
    inter_matrix = mutualism * mutualism_mat + competition * competition_mat

    # Set self-limitation
    np.fill_diagonal(inter_matrix, self_limitation)

    return inter_matrix


def create_network(inter_matrix, filepath0=CSV_FILEPATH,filepath=CSV_FILEPATH_2):
    df = pd.read_csv(filepath)  
    species=[]
    for specie in df["Specie"]:
        if specie not in species:
            species.append(specie)
    
    counter_prey = []
    df = pd.read_csv(filepath0, header=None)
    
    for i, row in df.iterrows():
        count=0
        for v in row:
            if v >0:
                count+=1
        counter_prey.append(count)
    
    #Initial population is the multiplication of how many predetors the specie has with 
    #the min number of population of a pray plus their min designated population
    
    G = nx.Graph()
    
    for new_node in range(len(species)):
        G.add_nodes_from([(new_node, {"specie": species[new_node], "population": (counter_prey[new_node]*20)+20})])
    
    for i in range(len(inter_matrix)):
        for j in range(i+1,len(inter_matrix)):
            if inter_matrix[i][j] > 0:
                #print(inter_matrix[i][j])
                G.add_edge(i, j, weight=inter_matrix[i][j])
            
    for node in G.nodes.data():
        print(node)
    
    pos = nx.spring_layout(G)
    names = nx.get_node_attributes(G, 'specie')
    nx.draw(G, pos, node_size=500, with_labels=True)
    #nx.draw_networkx_labels(G, pos, labels=names)
    plt.show()
    
    return G


#################### Main ####################
if __name__ == "__main__":
    inter_matrix = create_iteraction_matrix()
    
    G = create_network(inter_matrix)
    
    
