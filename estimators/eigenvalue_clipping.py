import numpy as np


class EigenvalueClipping:

    @staticmethod
    def get_mp_law_edges(c):
        lambda_lower = (1 - np.sqrt(c))**2
        lambda_upper = (1 + np.sqrt(c))**2
        return lambda_lower, lambda_upper

    @staticmethod
    def clip(scm, c):
        d = np.diag(np.diag(scm))
        corr_matrix = (d**(-1/2)) * scm * (d**(-1/2))
        lower_edge, upper_edge = EigenvalueClipping.get_mp_law_edges(c)
        eigenvalues, eigenvectors = np.linalg.eig(corr_matrix)
        bulk = [x for x in eigenvalues if x <= upper_edge]
        mean_bulk = np.mean(bulk)
        for i, value in enumerate(eigenvalues):
            if value <= upper_edge:
                eigenvalues[i] = mean_bulk
        clipped_corr = eigenvectors * eigenvalues * eigenvalues.T
        clipped_scm = (d**(1/2)) * clipped_corr * (d**(1/2))
        return clipped_scm


if __name__ == '__main__':

    import os
    parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    filename = "ASX data.csv"
    data_file = os.path.join(parent_dir, "data", filename)


