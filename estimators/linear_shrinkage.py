import numpy as np


class LinearShrinkage:

    @staticmethod
    def identity_target(scm):
        mu = np.mean(np.diag(scm))
        return mu * np.eye((scm.shape[0]))

    @staticmethod
    def svzc_target(scm):
        """
        Sample variance Zero Covariance Target:
        Diagonal elements contain sample variance (diagonal elements of SCM)
        Off-diagonal element replaced by zero (assuming no correlation between different features)
        """
        num_features = scm.shape[0]
        target = np.eye(num_features) * scm
        return target

    @staticmethod
    def svmc_target(scm):
        """
        Sample variance Mean Covariance Target:
        Diagonal elements contain sample variance (diagonal elements of SCM)
        Off-diagonal element replaced by mean covariance (mean of off-diagonal elements of SCM)
        """
        num_features = scm.shape[0]
        mean_covariance = np.sum(scm * ~ np.eye(num_features, dtype=bool)) / (num_features**2 - num_features)
        target = np.eye(num_features) * scm + ~np.eye(num_features, dtype=bool) * mean_covariance
        return target

    @staticmethod
    def get_target(scm, shrinkage_target):
        if shrinkage_target == "identity":
            target = LinearShrinkage.identity_target(scm)
        elif shrinkage_target == "svzc":
            target = LinearShrinkage.svzc_target(scm)
        elif shrinkage_target == "svmc":
            target = LinearShrinkage.svmc_target(scm)
        else:
            raise ValueError("Not a valid shrinkage target")
        return target
