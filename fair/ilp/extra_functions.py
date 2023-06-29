from numpy import ndim
from math import pow,log2,fabs

def normalize_array(array:list, min_val:float, max_val:float):
    """
    Normalizes the array using the minimum and maximum values on said array. The
    array must be a 1 dimension object and the values must be between the min and
    max, or an exception will be raised.

    The result will be a list of the same size with values between 0 and 1

    Parameters
    ----------
    array : list
        1 dimension object with a series of values
    min_val : float
        Minimum value posible in the array
    max_val : float
        Maximum value posible in the array
    """
    if ndim(array) != 1:
        raise Exception("The 'array' parameter must be a 1 dimension object")

    result = []

    for val in array:
        if val < min_val or max_val < val:
            raise Exception("Value ({}) of 'array' must be between the minimum ({}) and maximum ({}) value".format(val, min_val, max_val))
        result.append((val - min_val)/(max_val - min_val))

    return result

def pos_bias(pos: int, k: int, prob: float) -> float:
    """
    Receives a position and calculates the bias of said position. It will calculate
    only for the first k subjects, after k, the bias will be 0.

    Parameters
    ----------
    pos : integer
        Position
    k : integer
        Amount of top subjects to consider for calculations
    prob : float
        Probability that any subject will be chosen
    """
    if pos > k:
        return 0
    else:
        return prob * pow(1-prob, pos-1)
        

def l1_norm_dist(dist_A:float,dist_R:float) -> float:
    """
    Calculates the L1-norm distance between the empirical distributions A and R.
    This function is intended to be used alongside a for loop to simulate the
    sumatory for the full L1-norm distance.

    >>> for element in range(qty_elements):
    >>>     sum += l1_norm_dist(a[element], r[element])

    Parameters
    ----------
    dist_A : float
        Empirical distribution A (attention)
    dist_R : float
        Empirical distribution R (relevance)
    """
    return fabs(dist_A - (dist_R))

def dcg_score(value:float,pos:int) -> float:
    """
    Calculates the DCG score for a value in a position. This function is intended
    to be used alongside a for loop to simulate the sumatory for the DCG@k score.

    >>> for value in relevant_values:
    >>>     sum += dcg_score(value)

    Parameters
    ----------
    value : float
        Value to calculate the DCG score
    pos : int
        Position to calculate the DCG score
    """
    return (pow(2,value)-1)/log2(pos+1)

def fair_share_attention(dist_Ai:float, dist_Ri:float, score_i:float) -> float:
    """
    Calculates the fair share attention as described in [1].

    Parameters
    ----------
    dist_A : float
        Empirical distribution A (attention) of subject i
    dist_R : float
        Empirical distribution R (relevance) of subject i
    score : float
        Relevance score of subject i

    [1] Asia J. Biega, Krishna P. Gummadi, and Gerhard Weikum. 2018.
    Equity of Attention: Amortizing Individual Fairness in Rankings.
    """
    return dist_Ai - (dist_Ri + score_i)