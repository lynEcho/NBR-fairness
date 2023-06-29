from scipy.stats import binom


def avg_prefix(rec_df, group):
    """
    FA*IR - average of prefix probabilities.

    Args:
        rec_df(pandas.DataFrame):
            Ranking lists. Must be for a single user, sorted by increasing rank.
        group(GroupInfo): the groupinfo object.
    """
    # using the current rank as prefix
    k = rec_df['rank']
    # calculate number of protected group up to said prefix
    Tp = rec_df[group.minor].cumsum()
    # calculate prefix probability; return the average
    probs = binom.cdf(Tp, k, group.tgt_p_binomial)
    return probs.mean()
