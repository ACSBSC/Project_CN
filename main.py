#################### Imports ####################
import numpy as np
import os
import networkx as nx
import matplotlib.pyplot as plt
import random

#################### Settings ####################
MUTUALISMS_FILEPATH = os.path.join("FW_005", "FW_005.csv")
SPECIES_FILEPATH = os.path.join("FW_005", "FW_005_species.csv")
SELF_LIMITATION = 0.01
MUTUALISM = 1e-5
COMPETITION = 1e-6

MIN_POPULATION = 1e2
POPULATION_PER_PREY = 1e2


#################### Functions ####################
def create_iteractions_matrix(mutualism_filepath=MUTUALISMS_FILEPATH, self_limitation=SELF_LIMITATION,
                              mutualism=MUTUALISM, competition=COMPETITION):
    # Read mutualism matrix
    mutualism_mat = np.genfromtxt(mutualism_filepath, delimiter=',', dtype=float)
    num_individuals = mutualism_mat.shape[0]

    # Remove cannibalism
    np.fill_diagonal(mutualism_mat, 0)

    # Make columns to sum 1
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

    return inter_matrix, mutualism_mat, competition_mat


def balanced_ini_state(interactions_mat, mutualism_mat, min_population=MIN_POPULATION,
                       population_per_prey=POPULATION_PER_PREY):
    # Get number of preys per specie
    num_preys_arr = np.count_nonzero(mutualism_mat, axis=1)

    # Initial population is the multiplication of how many predators the specie has with
    # the min number of population of a pray plus their min designated population
    populations = num_preys_arr * population_per_prey + min_population
    populations = populations.astype(float)

    # Self growing factor is defined for a balanced state, so is equal ΣMij*xj
    self_growings = np.sum(interactions_mat * populations, axis=1)

    return populations, self_growings


def create_network(interactions_mat, populations, self_growings, species_filepath=SPECIES_FILEPATH, verbose=True):
    G = nx.Graph()

    # Get species names
    species = np.genfromtxt(species_filepath, delimiter=',', dtype=str)

    # Create nodes nodes with the corresponding properties
    for node_idx, (specie, population, self_growing) in enumerate(zip(species, populations, self_growings)):
        G.add_nodes_from([(node_idx, {"specie": specie, "ini_population": population, "self_growing": self_growing})])

    # Create edges
    for i in range(len(interactions_mat)):
        for j in range(i + 1, len(interactions_mat)):
            if interactions_mat[i][j] > 0:
                G.add_edge(i, j, weight=interactions_mat[i][j])

    if verbose:
        # Print nodes data
        for node in G.nodes.data():
            print(node)

    return G


def plot_network(G):
    title = 'Gulf of Cadiz Food-Web Network'
    plt.figure(figsize=(10, 5))
    ax = plt.gca()
    ax.set_title(title)
    pos = nx.spring_layout(G)
    names = nx.get_node_attributes(G, 'specie')
    nx.draw(G, pos, node_size=500, with_labels=True, ax=ax)
    # nx.draw_networkx_labels(G, pos, labels=names)
    plt.show()


def simulation(interactions_mat, self_growings, populations, G, first_id_to_extinct, n_iterations):
    # Initialize populations history
    populations_history = np.empty((n_iterations + 1, len(populations)))
    populations_history[0, :] = populations[:]

    # Extinct the first
    populations_history[0, first_id_to_extinct] = 0
    extinct_node = G.nodes[first_id_to_extinct]
    extinct_dict = {first_id_to_extinct: extinct_node}
    print(f"FIRST TO EXTINCT: ID{first_id_to_extinct} = {extinct_node}")

    # Simulation loop
    performed_iterations = n_iterations
    for i in range(1, n_iterations + 1):
        # Compute and apply update
        update = populations_history[i - 1] * (
                    self_growings - np.sum(interactions_mat * populations_history[i - 1], axis=1))
        populations_history[i] = populations_history[i - 1] + update

        # Check for extinction
        extinct_ids = np.where(populations_history[i] <= 0)[0]
        if len(extinct_ids) > 0:
            populations_history[i, extinct_ids] = 0
            # Print extinction information
            for id_extinct in extinct_ids:
                if not id_extinct in extinct_dict:
                    extinct_dict[id_extinct] = (i, G.nodes[id_extinct])
                    print(f"Iteration {i} | EXTINCTION: ID{id_extinct} = {G.nodes[id_extinct]}")

        # Check for end of simulation
        if np.sum(update) == 0:
            performed_iterations = i
            populations_history = populations_history[:i + 1, :]
            break

    print(f"SIMULATION FINISHED AFTER {performed_iterations} iterations")

    return extinct_dict, populations_history


#################### Main ####################
if __name__ == "__main__":
    # Interactions matrix
    interactions_mat, mutualism_mat, competition_mat = create_iteractions_matrix()

    # Simulation parameters for ecosystem equilibrium
    populations, self_growings = balanced_ini_state(interactions_mat, mutualism_mat)

    # Network
    G = create_network(interactions_mat, populations, self_growings)

    print("------------------------------------------------------")

    n_iterations = 1000
    first_id_to_extinct = 28  # 18 = Dolphins, 28 = Macrozooplankton, 38 = Sharks # random.randint(0, len(populations))
    extinct_dict, populations_history = simulation(interactions_mat, self_growings, populations, G, first_id_to_extinct,
                                                   n_iterations)
    print("------------------------------------------------------")

    # Manage simulation results
    new_populations = populations_history[-1, :]
    survival_ids = np.where(new_populations > 0)[0]
    print(f"{len(survival_ids)} ANIMALS SURVIVED")
    for survived_id in survival_ids:
        print(f"SURVIVED: ID{survived_id} = {G.nodes[survived_id]} "
              f"with a population of {int(new_populations[survived_id])})")
