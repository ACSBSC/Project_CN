#################### Imports ####################
import numpy as np
import os
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import random

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
MUTUALISMS_FILEPATH = os.path.join("FW_005", "FW_005.csv")
SPECIES_FILEPATH = os.path.join("FW_005", "FW_005_species.csv")
SELF_LIMITATION = 0.01
MUTUALISM = 1e-5
COMPETITION = 1e-6

MIN_POPULATION = 1e2
POPULATION_PER_PREY = 1e3


#################### Functions ####################
def create_iteractions_matrix(mutualism_filepath=MUTUALISMS_FILEPATH, self_limitation=SELF_LIMITATION,
                             mutualism=MUTUALISM, competition=COMPETITION):
    # Read mutualism matrix
    mutualism_mat = np.genfromtxt(mutualism_filepath, delimiter=',', dtype=float)
    num_individuals = mutualism_mat.shape[0]

    # Remove cannibalism
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

def create_network(inter_matrix, mutualism_filepath=MUTUALISMS_FILEPATH,
                   species_filepath=SPECIES_FILEPATH, min_population=MIN_POPULATION,
                   population_per_prey=POPULATION_PER_PREY):
    # Get species names
    species = np.genfromtxt(species_filepath, delimiter=',', dtype=str)

    # Get number of preys per specie
    mutualism_mat = np.genfromtxt(mutualism_filepath, delimiter=',', dtype=float)
    np.fill_diagonal(mutualism_mat, 0)  # Remove cannibalism
    num_preys_arr = np.count_nonzero(mutualism_mat, axis=1)

    # Initial population is the multiplication of how many predators the specie has with
    # the min number of population of a pray plus their min designated population
    populations = num_preys_arr * population_per_prey + min_population
    populations = populations.astype(float)

    # Self growing factor is defined for an stable state, so is equal ΣMij*xj
    self_growings = np.sum(inter_matrix * populations, axis=1)

    # Create network nodes with the corresponding properties
    G = nx.Graph()
    for node_idx, (specie, population, self_growing) in enumerate(zip(species, populations, self_growings)):
        G.add_nodes_from([(node_idx, {"specie": specie, "ini_population": population, "self_growing": self_growing})])

    # Create network edges
    for i in range(len(inter_matrix)):
        for j in range(i + 1, len(inter_matrix)):
            if inter_matrix[i][j] > 0:
                G.add_edge(i, j, weight=inter_matrix[i][j])

    # Print nodes data
    for node in G.nodes.data():
        print(node)

    # Plot network
    pos = nx.spring_layout(G)
    names = nx.get_node_attributes(G, 'specie')
    nx.draw(G, pos, node_size=500, with_labels=True)
    # nx.draw_networkx_labels(G, pos, labels=names)
    plt.show()

    return G, populations, self_growings

def grow_species(populations, self_growings, inter_matrix):
    update = populations * (self_growings - np.sum(inter_matrix * populations, axis=1))
    return update

#################### Main ####################
if __name__ == "__main__":
    inter_matrix = create_iteractions_matrix()

    G, populations, self_growings = create_network(inter_matrix)

    # Initialize simulation
    iterations = [1000]
    print("------------------------------------------------------")

    # Remove an specie
    id_removed = 18  # random.randint(0, 43)
    populations[id_removed] = 0
    extinct_dict = {id_removed: G.nodes[id_removed]}
    print(f"FIRST TO EXTINCT: {id_removed} = {G.nodes[id_removed]}")

    for it in iterations:
        for i in range(it):
            update = grow_species(populations, self_growings, inter_matrix)
            populations += update

            # Check for extinction
            extinct_ids = np.where(populations <= 0)[0]
            if len(extinct_ids) > 0:
                populations[extinct_ids] = 0
                for id_extinct in extinct_ids:
                    if not id_extinct in extinct_dict:
                        extinct_dict[id_extinct] = G.nodes[id_extinct]
                        print(f"Iteration {i+1} | EXTINCTION: {id_extinct} = {G.nodes[id_extinct]}")

            # Check for end of simulation
            if len(extinct_ids) == len(populations):
                break

    print("--------------------------------------------------------")
    survival_ids = np.where(populations > 0)[0]
    for survived_id in survival_ids:
        print(f"SURVIVED: {survived_id} = {G.nodes[survived_id]['specie']} with a population of {int(populations[survived_id])} (initial {int(G.nodes[survived_id]['ini_population'])})")
