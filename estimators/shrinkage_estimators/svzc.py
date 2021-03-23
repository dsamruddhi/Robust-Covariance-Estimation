import numpy as np


class SVZCTarget:

    """
    Sample variance Zero Covariance Target:
    Diagonal elements contain sample variance (diagonal elements of SCM)
    Off-diagonal element replaced by zero (assuming no correlation between different features)
    """

    @staticmethod
    def get_target(scm):
        num_features = scm.shape[0]
        target = np.eye(num_features) * scm
        return target


if __name__ == '__main__':

    scm = np.ones((10, 10), dtype=int)
    scm[4, 4] = 2
    print(SVZCTarget.get_target(scm))
