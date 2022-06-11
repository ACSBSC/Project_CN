######################################################################
#   Description: Main script of the final Complex Networks project   #
#   Authors:                                                         #
#   * Anne Constanze Schreiber Brito                                 #
#   * Ruben Vera García                                              #
#   * Benet Manzanares Salor                                         #
######################################################################



######################################## Imports ########################################
import numpy as np
import os
import networkx as nx
import matplotlib.pyplot as plt
import math

######################################## Settings ########################################
MUTUALISMS_FILEPATH = os.path.join("FW_005", "FW_005.csv")
SPECIES_FILEPATH = os.path.join("FW_005", "FW_005_species.csv")
SELF_LIMITATION = 1e-9  # Probability population dies
MUTUALISM_WEIGHT = 1e-4
COMPETITION_WEIGHT = 1e-9

SELF_LIMITATIONS = [math.pow(10, -i) for i in range(1, 10)]
MUTUALISM_WEIGHTS = [math.pow(10, -i) for i in range(1, 10)]
COMPETITION_WEIGHTS = [math.pow(10, -i) for i in range(1, 10)]
# BEST COMBINATION (22): 1.000000e-09 | 1.000000e-04 | 1.000000e-09

MIN_POPULATION = 1e2
POPULATION_INC_PER_PREY = 1e3

N_ITERATIONS = 100
EXTICTION_FACTOR = 0


######################################## Functions ########################################

########### Interactions matrix ###########
def get_mutualism_and_competition_matrices(predation_mat):
    # Read mutualism matrix
    num_individuals = predation_mat.shape[0]

    # Remove cannibalism
    np.fill_diagonal(predation_mat, 0)

    # Make columns to sum 1
    col_weights = predation_mat.sum(0)
    np.divide(predation_mat, col_weights, where=col_weights != 0, out=predation_mat)

    # Compute mutualism matrix considering being eaten and eat
    mutualism_mat = np.empty_like(predation_mat)
    for i in range(num_individuals):
        for j in range(num_individuals):
            mutualism_mat[i, j] = predation_mat[i, j] - predation_mat[j, i]

    # Compute competition matrix
    competition_mat = np.zeros_like(mutualism_mat)
    for i in range(num_individuals):
        for j in range(num_individuals):
            if i != j:
                competition_mat[i, j] = np.sum(predation_mat[:, j], where=predation_mat[:, i] > 0)

    return mutualism_mat, competition_mat


def create_iteractions_matrix(mutualism_mat, competition_mat, self_limitation=SELF_LIMITATION,
                              mutualism_weight=MUTUALISM_WEIGHT, competition_weight=COMPETITION_WEIGHT):
    # Create interactions matrix
    inter_matrix = mutualism_weight * mutualism_mat + competition_weight * competition_mat

    # Set self-limitation
    np.fill_diagonal(inter_matrix, self_limitation)

    return inter_matrix


########### Parameters selection ###########
def ini_populations(predation_mat, min_population=MIN_POPULATION, population_per_prey=POPULATION_INC_PER_PREY):
    # Get number of preys per specie
    num_preys_arr = np.sum(predation_mat, axis=1)

    # Initial population is the multiplication of how many predators the specie has with
    # the min number of population of a pray plus their min designated population
    populations = num_preys_arr * population_per_prey + min_population
    populations = populations.astype(float)

    return populations


def balanced_ini_state(interactions_mat, populations):
    # Self growing factor is defined for a balanced state, so is equal ΣMij*xj
    self_growings = np.sum(interactions_mat * populations, axis=1)

    return self_growings


def parameters_search(predation_mat, mutualism_mat, competition_mat):
    best_value = float("-inf")
    best_self_limitation = 0
    best_mutualism_w = 0
    best_competition_w = 0

    # Grid search
    for self_limitation in SELF_LIMITATIONS:
        for mutualism_w in MUTUALISM_WEIGHTS:
            for competition_w in COMPETITION_WEIGHTS:
                populations_history_list = test(predation_mat, mutualism_mat, competition_mat,
                                                self_limitation, mutualism_w, competition_w)

                extinction_amounts = get_extinctions_amounts(populations_history_list)
                current_value = len(np.unique(extinction_amounts))
                if current_value > best_value:
                    print("NEW BEST: ", current_value, self_limitation, mutualism_w, competition_w)
                    best_value = current_value
                    best_self_limitation = self_limitation
                    best_mutualism_w = mutualism_w
                    best_competition_w = competition_w

    return best_value, best_self_limitation, best_mutualism_w, best_competition_w


########### Network ###########
def create_network(interactions_mat, populations, self_growings, species_filepath=SPECIES_FILEPATH, verbose=True):
    G = nx.Graph()

    # Get species names
    species = np.genfromtxt(species_filepath, delimiter=',', dtype=str)

    # Create nodes with the corresponding properties
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


########### Simulation ###########
def simulation(interactions_mat, self_growings, populations, first_id_to_extinct, n_iterations=N_ITERATIONS,
               extinction_factor=EXTICTION_FACTOR):
    # Initialize populations history
    populations_history = np.empty((len(populations), n_iterations + 1))
    populations_history[:, 0] = populations[:]

    # Extinct the first
    populations_history[first_id_to_extinct, 0] *= extinction_factor

    # Simulation loop
    for i in range(1, n_iterations + 1):
        # Compute and apply update
        update = populations_history[:, i - 1] * (
                self_growings - np.sum(interactions_mat * populations_history[:, i - 1], axis=1))
        populations_history[:, i] = populations_history[:, i - 1] + update

        # Check for extinction
        extinct_ids = np.where(populations_history[:, i] <= 0)[0]
        if len(extinct_ids) > 0:
            populations_history[extinct_ids, i] = 0

        # Check for end of simulation
        if np.sum(update) == 0:
            populations_history = populations_history[:, :i + 1]
            break

    return populations_history


def test(predation_mat, mutualism_mat, competition_mat, self_limitation, mutualism_w, competition_w, verbose=False):
    # Interactions matrix
    interactions_mat = create_iteractions_matrix(mutualism_mat, competition_mat,
                                                 self_limitation, mutualism_w, competition_w)

    # Define others parameters
    populations = ini_populations(predation_mat)
    self_growings = balanced_ini_state(interactions_mat, populations)

    # Network
    G = create_network(interactions_mat, populations, self_growings, verbose=False)
    # plot_network(G)
    # print("------------------------------------------------------")

    populations_history_list = []
    for first_to_extinct in range(len(populations)):
        populations_history = simulation(interactions_mat, self_growings, populations, first_to_extinct)
        populations_history_list.append(populations_history)

        # Show results if desired
        if verbose:
            print_simulation_results(populations_history, G, first_to_extinct)
            plot_populations_history(populations_history, G, first_to_extinct, include_extinct=True)
            plot_populations_history(populations_history, G, first_to_extinct, include_extinct=False)

    return populations_history_list


########### Results ###########
def print_simulation_results(populations_history, G, first_to_extinct):
    #print(f"SIMULATION FINISHED AFTER {len(populations_history[0, :])-1} iterations")

    # Who is extinct who survives
    num_survived = 0
    for specie_id, history in enumerate(populations_history):
        node = G.nodes[specie_id]
        if history[-1] == 0:
            extinction_iteration = np.argmax(history == 0)
            """if extinction_iteration == 0:
                print(f"FIRST TO EXTINCT: ID{specie_id} = {node}")
            else:
                print(f"EXTINCTION: ID{specie_id} = {node} at iteration {extinction_iteration}")"""
        else:
            num_survived += 1
            #print(f"SURVIVED: ID{specie_id} = {node} "
            #      f"with a population of {int(history[-1])})")

    print(f"WHEN {G.nodes[first_to_extinct]['specie']} EXTINCTS, {num_survived} SPECIES SURVIVE")


def plot_populations_history(populations_history, G, first_to_extinct, include_extinct):
    plt.figure(figsize=(15, 7))

    # Set title
    specie_name = G.nodes[first_to_extinct]["specie"]
    include_extinct_str = "" if include_extinct else "NOT "
    include_extinct_str += "including extinct"
    plt.suptitle(f'Populations history for {specie_name} extinction {include_extinct_str}')

    # Set plots
    markers = ('o', 'v', '^', '<', '>', '1', 's', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X', 'x', '|', 0, 2, 4, 5, 6, 7)
    ploted_idx = 0
    for idx, population_history in enumerate(populations_history):
        has_survived = population_history[-1] > 0
        if include_extinct or has_survived:
            label = G.nodes[idx]["specie"]
            label += "" if has_survived else " (extinct)"
            plt.plot(population_history, label=label,
                     marker=markers[ploted_idx % len(markers)])
            ploted_idx += 1

    # Final settings and show
    plt.legend(prop={'size': 8})
    plt.xlabel("#Iterations")
    plt.ylabel("Population size")
    plt.show()


def get_extinctions_amounts(populations_histories) -> np.array:
    extinction_amounts = [np.count_nonzero(history[:, -1]) for history in populations_histories]
    extinction_amounts = np.array(extinction_amounts)
    return extinction_amounts



#################### Main ####################
if __name__ == "__main__":
    # Compute mutualism and competition matrices
    predation_mat = np.genfromtxt(MUTUALISMS_FILEPATH, delimiter=',', dtype=float)
    mutualism_mat, competition_mat = get_mutualism_and_competition_matrices(predation_mat)

    # Search best configuration
    """best_value, best_self_limitation, best_mutualism_w, best_competition_w = parameters_search(predation_mat,
                                                                                               mutualism_mat,
                                                                                               competition_mat)
    print(f"BEST COMBINATION (Value={best_value}): Self-limitation={best_self_limitation:e} |"
          f" Mutualism weight={best_mutualism_w:e} | Competition weight={best_competition_w:e}")"""

    # Experiment
    populations_history_list = test(predation_mat, mutualism_mat, competition_mat,
                                    SELF_LIMITATION, MUTUALISM_WEIGHT, COMPETITION_WEIGHT, verbose=False)
    extinction_amounts = get_extinctions_amounts(populations_history_list)
    print(extinction_amounts)
