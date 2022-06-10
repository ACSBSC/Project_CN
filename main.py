#################### Imports ####################
import numpy as np
import os
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import random


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

def initialization(inter_matrix, mutualism_filepath=MUTUALISMS_FILEPATH,
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
    title = 'Gulf of Cadiz Food-Web Network'
    plt.figure(figsize=(10,5))
    ax = plt.gca()
    ax.set_title(title)
    pos = nx.spring_layout(G)
    names = nx.get_node_attributes(G, 'specie')
    nx.draw(G, pos, node_size=500, with_labels=True, ax=ax)
    # nx.draw_networkx_labels(G, pos, labels=names)
    plt.show()

    return G, populations, self_growings



def simulation(inter_matrix, self_growings, populations, G, first_id_to_extinct, iterations):
    populations[first_id_to_extinct] = 0
    extinct_node = G.nodes[first_id_to_extinct]
    extinct_dict = {first_id_to_extinct: extinct_node}
    print(f"FIRST TO EXTINCT: ID{first_id_to_extinct} = {extinct_node}")

    for i in range(iterations):
        # Compute and apply update
        update = populations * (self_growings - np.sum(inter_matrix * populations, axis=1))
        new_populations = populations + update

        # Check for extinction
        extinct_ids = np.where(new_populations <= 0)[0]
        if len(extinct_ids) > 0:
            new_populations[extinct_ids] = 0
            # Print extinction information
            for id_extinct in extinct_ids:
                if not id_extinct in extinct_dict:
                    extinct_dict[id_extinct] = (i, G.nodes[id_extinct])
                    print(f"Iteration {i + 1} | EXTINCTION: ID{id_extinct} = {G.nodes[id_extinct]}")

        # Check for end of simulation
        if np.sum(new_populations - populations) == 0:
            break
        else:
            populations = new_populations

    return extinct_dict, populations


#################### Main ####################
if __name__ == "__main__":
    inter_matrix = create_iteractions_matrix()

    G, populations, self_growings = initialization(inter_matrix)
    print("------------------------------------------------------")

    iterations = 1000
    first_id_to_extinct = 28    # 18 = Dolphins
    extinct_dict, new_populations = simulation(inter_matrix, self_growings, populations, G, first_id_to_extinct, iterations)
    print("------------------------------------------------------")

    survival_ids = np.where(new_populations > 0)[0]
    for survived_id in survival_ids:
        print(f"SURVIVED: ID{survived_id} = {G.nodes[survived_id]} with a population of {int(new_populations[survived_id])})")
