import pandas as pd
import numpy as np


def estimate_p(matrix):
    """
    P-Hat Estimator

    Args:
        matrix (Numpy Matrix): Columns are groups in
        a user reccommendation dataframe

    Returns:
        [Numpy Array]: [N x 1 Matrix of groups average]
    """
    p_hat = np.sum(matrix, axis=0) / len(matrix)
    return p_hat


def awrf(rec_df, group, weight_vector, p_hat=None):
    """
    Attention-Weighted Ranked Fairness, with binomial distance function.

    Original Metric: Distance

    Related Paper: "Quantifying the Impact of User Attention on Fair Group
                    Representation in Ranked Lists"
    First Author:   Piotr Sapiezynski

    Args:
        rec_df (pandas.DataFrame): A list of recommendations for a single user
        group (GroupInfo Object): See groupinfo.py; manages groups in rec_df
        weight_vector (position based object):
            Weights for this user's ranks.
        p_hat(float or None):
            The target probability, or ``None`` to use the underlying corpus as
            the target.

    Returns:
        distance (pandas.Series): Distribution of exposure over a list of items
                                  recommendeded to a single user
    """
    # FIXME does not work for more than 2 groups!
    lr = pd.DataFrame(rec_df[[group.minor, group.major, group.unknown]])
    #lr = pd.DataFrame(rec_df[[group.minor, group.major]])
    matrix = np.asarray(lr.copy())
    if p_hat is None:
        p_hat = group.tgt_p_binomial
    # normalize weight vector to a distribution
    weight_vector = weight_vector / weight_vector.sum()
    E_R = np.matmul(weight_vector.transpose(), matrix)
    E_R = pd.Series(E_R, index =[group.minor, group.major, group.unknown])
    #E_R = pd.Series(E_R, index =[group.minor, group.major])
    min_prob = E_R[group.minor] / (E_R[group.minor] + E_R[group.major])
    # we want the difference between major and target probability
    return np.abs(min_prob - p_hat)
