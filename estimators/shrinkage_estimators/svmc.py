import numpy as np


class SVMCTarget:

    """
    Sample variance Mean Covariance Target:
    Diagonal elements contain sample variance (diagonal elements of SCM)
    Off-diagonal element replaced by mean covariance (mean of off-diagonal elements of SCM)
    """

    @staticmethod
    def get_target(scm):
        num_features = scm.shape[0]
        mean_covariance = np.sum(scm * ~ np.eye(num_features, dtype=bool)) / (num_features**2 - num_features)
        target = np.eye(num_features) * scm + ~np.eye(num_features, dtype=bool) * mean_covariance
        return target


if __name__ == '__main__':

    scm = np.ones((10, 10), dtype=int)
    scm[2, 3] = 3
    print(SVMCTarget.get_target(scm))
