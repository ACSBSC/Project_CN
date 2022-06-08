#################### Imports ####################
import numpy as np
import os

#################### Settings ####################
CSV_FILEPATH = os.path.join("FW_005", "FW_005.csv")
SELF_LIMITATION = 0.1
MUTUALISM = 0.1
COMPETITION = 0.001


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


#################### Main ####################
if __name__ == "__main__":
    inter_matrix = create_iteraction_matrix()
