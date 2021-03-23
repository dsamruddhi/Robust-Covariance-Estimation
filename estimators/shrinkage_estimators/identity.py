import numpy as np


class IdentityTarget:

    @staticmethod
    def get_target(scm):
        return np.eye((scm.shape[0]))


if __name__ == '__main__':

    scm = np.ones((10, 10), dtype=int)
    print(IdentityTarget.get_target(scm))
