#!/usr/bin/env python3

# Imports
from math import inf
from numpy import array, concatenate
from numpy.core.fromnumeric import argsort, ndim, shape, sort
from pulp.pulp import LpProblem, LpVariable, lpSum
from pulp.utilities import value
from pulp.constants import LpMinimize, LpStatus
import sys
from ilp.extra_functions import normalize_array, pos_bias, l1_norm_dist, dcg_score, fair_share_attention

LOWER = 0
UPPER = 1


class ILPBased:
    """ILPBased method for Equity of Attention [1]

    Creates a wrapper-like class containing a set of rankings, which are arrays of relevance
    scores for a number of subjects, and the methods to optimize the order of each subject with
    the objective of achieving Equity of Attention. This concept is defined as "Ranked subjects
    receive attention that is proportional to their worthiness in a given search task."
    
    This class contains an array of rankings, with each of them normalized (in values from 0 to 1). 
    For this normalization to happen, the minimum and maximum possible values in each ranking are
    needed. This values are given through the 'bounds' argument, which must be a pair of values or
    an array of pairs of values, depending on the amount of rankings in the parameter 'rank_series'.

    [1] Asia J. Biega, Krishna P. Gummadi, and Gerhard Weikum. 2018.
    Equity of Attention: Amortizing Individual Fairness in Rankings.

    """

    def __init__(
        self: "ILPBased",
        rank_series: object,
        bounds: object,
    ):
        """ILP Based Equity of Attention Model Constructor

        Args:
            ranking_series (object): A ranking or an array of rankings.
                A ranking is a collection of relevance scores, with each index corresponding to a
                subject. If an array of rankings is passed, the relevance scores for each subject
                must be on the same index for every ranking.
            bounds (object): A pair or an array of pairs of values.
                Each pair corresponds to one ranking in order. Each index on bounds must correspond
                to a ranking in rankings. If only one ranking is passed, only a pair of values is
                needed.
        """

        if ndim(rank_series) == 1:
            if ndim(bounds) == 1 and len(bounds) == 2:
                self.rank_series = array([
                    normalize_array(
                        rank_series,
                        bounds[LOWER],
                        bounds[UPPER],
                    )
                ])
            else:
                raise Exception(
                    "For singular ranking input, bounds must be a pair of values indicating the lower and upper bounds"
                )
        elif ndim(rank_series) == 2:
            if ndim(bounds) == 2 and shape(bounds) == (len(rank_series), 2):
                self.rank_series = array([
                    normalize_array(
                        rank_series[rank],
                        bounds[rank][LOWER],
                        bounds[rank][UPPER],
                    )
                    for rank in range(len(rank_series))
                ])
            else:
                raise Exception(
                    "For multi-ranking input, the amount of pairs on bounds must match the amount of rankings"
                )
        else:
            raise Exception(
                "Only a ranking or a series of rankings are valid input"
            )

        self.rank_qty = len(self.rank_series) #number of users
        self.subj_qty = len(self.rank_series[0]) #number of items
        self.pref_qty = 100 #pre-filter t items
        self.pos = [0] * self.rank_qty
        self.pref = None
        self.rel_subj = [0] * self.rank_qty
        self.final = [0] * self.rank_qty #final rankings
        self.rerank_rel = [0] * self.rank_qty #final relevance
        self.acumm_A = [0] * self.subj_qty
        self.acumm_R = [0] * self.subj_qty
        self.ideal_score = [0] * self.rank_qty
        self.k = 20 #top-k relevance scores
        self.prob = 0.5 #position bias
        self.theta = 0.9

        self.unfairness_vals = []

        self.problem = None
        self.decisions = None

    def __pos_bias(
        self: "ILPBased",
        pos: int,
    ):
        """Returns the position bias
        """
        return pos_bias(
            pos,
            self.k,
            self.prob
        )

    def __unfairness(
        self: "ILPBased",
        rank: int,
        i: int,
        j: int,
        var: int,
    ):
        """Returns the unfairness value of the prefilter subject i in position j.
        
        This value is multiplied by the variable[i][j] of the ILP.
        """
        subj = self.pref[i]
        pos = j + 1
        return var * l1_norm_dist(
            self.acumm_A[subj] + self.__pos_bias(pos),
            self.acumm_R[subj] + self.rank_series[rank][subj],
        ) #(3)min

    def __quality(
        self: "ILPBased",
        rank: int,
        i: int,
        j: int,
        var: int,
    ):
        """Returns the ranking quality value of the prefilter subject i in position j
    
        This value is multiplied by the variable[i][j] of the ILP.
        """
        subj = self.pref[i]
        pos = j + 1
        return var * dcg_score(
            self.rank_series[rank][subj],
            pos,
        ) #(3)subject

    def __fairshare(
        self: "ILPBased",
        rank: int,
        subj: int,
    ):
        """Returns the fairshare attention value for the subject

        This value will be infinity if is a relevant subject.
        """
        if subj in self.rel_subj[rank]:
            return inf
        else:
            return fair_share_attention(
                self.acumm_A[subj],
                self.acumm_R[subj],
                self.rank_series[rank][subj]
            )

    def __iterate(
        self: "ILPBased",
        rank: int,
    ):
        """Iteration of the ILP model
        
        This is called by the 'start' method for the optimization of each prefilter subject in
        each ranking for a number of iterations. The prefilter of the subjects are done in the
        'start' method.
        """
        ROWS = COLS = N = range(self.pref_qty) #t
        K = range(self.k)
        threshold = self.theta * self.ideal_score[rank] #eq.(3)

        # Problem and choices creation
        problem = LpProblem("ILPBased", LpMinimize)
        choices = LpVariable.dicts("Choices", (ROWS, COLS), cat="Binary")

        # Objective function to minimize
        problem += lpSum([
            self.__unfairness(rank, i, j, choices[i][j])
            for j in N
            for i in N
        ])

        # Constraint 1: Maintain the ranking quality above the threshold
        problem += lpSum([
            self.__quality(rank, i, j, choices[i][j])
            for i in N
            for j in K
        ]) >= threshold

        # Constraint 2: Every row must sum to 1
        for i in N:
            problem += lpSum([
                choices[i][j]
                for j in N
            ]) == 1

        # Constraint 3: Every row must sum to 1
        for j in N:
            problem += lpSum([
                choices[i][j]
                for i in N
            ]) == 1

        problem.solve()

        return (problem.status, choices)

    def prepare(
        self: "ILPBased",
        pref_qty: int,
        k: int,
        prob: float,
        theta: float,
    ):
        """Prepare the attributes for optimization

        Args:
            pref_qty (int): Quantity of subjects to prefilter in each iteration
            k (int): Quantity of relevant subjects to be considered in some calculations
            prob (float): Probability of each subjects of being chosen
            theta (float): Threshold for the ideal DCG score of each ranking
            solver (string): Name of the solver to be used for optimization.
                Options are: CBC for COIN Branch and Cut, GRB for Gurobi
        """

        # Save parameters into attributes
        self.pref_qty = min(pref_qty, self.subj_qty) #t
        self.k = min(k, self.pref_qty)
        self.prob = min(max(prob, 0), 1)
        self.theta = min(max(theta, 0), 1)

        self.pos = [0] * self.rank_qty #number of users
        self.rel_subj = [0] * self.rank_qty
        self.ideal_score = [0] * self.rank_qty

        for r in range(self.rank_qty):
            # Temp vars for position and relevant subjects for this ranking
            r_pos = [0] * self.subj_qty
            r_rel_subj = [0] * self.k

            # Create the lists for the positions and relevant subjects
            sorted_rank = argsort(-self.rank_series[r]) #return index in descending, [2579 1840  472 ... 7163 7415 5896]
    
            for pos, index in enumerate(sorted_rank): #0, 2579
    
                r_pos[index] = pos #position list
                if pos < self.k:
                    r_rel_subj[pos] = index #top-k index

            self.pos[r] = r_pos
            self.rel_subj[r] = r_rel_subj

            # Get the ideal score of the ranking, only top-k
            self.ideal_score[r] = sum([
                dcg_score(val, j+1)
                for j, val in enumerate(self.rank_series[r][r_rel_subj])
            ]) #dcg@k(rou)

    def start(
        self: "ILPBased",
        iters: int,
    ):
        """Start the optimization process

        This will proceed for the number of iterations passed for each ranking in order.
        At the end of each iteration, the unfairness value will be calculated. This value
        is given by the L1 Norm distance of all the acummulated attention and relevance by
        the subjects.

        Args:
            iters (int): Number of iterations to optimize each ranking
        """

        ROWS = COLS = range(self.pref_qty) #t

        # Loop of iterations
        for it in range(iters):
            print("="*60)
            print("Iteration {}".format(it))

            print("Optimizing", end="")
            # Loop of rankings
            for rank in range(self.rank_qty): #for each user
                print(".", end="")

                # Prefilter the ranking, top-t id
                low_fairshare_subj = argsort([
                    self.__fairshare(rank, s)
                    for s in range(self.pref_qty)
                ])[:self.pref_qty - self.k]

                self.pref = concatenate(
                    (self.rel_subj[rank], low_fairshare_subj)
                )
                
                # Iterate and get the results
                solution = self.__iterate(rank) #return (problem.status, choices)
                solution_status = LpStatus[solution[0]]
                solution_vars = solution[1]

                if solution_status != "Optimal": #Optimal solution exists and is found.
                    raise Exception(
                        "Could not optimize on iteration {} for ranking {}, Pulp returned {}".format(
                            it, rank, solution_status)
                    )

                # Temp vars
                result = [0] * self.pref_qty #t
                result_pos = [0] * self.pref_qty
                sorted_pos = -sort(
                    -array([
                        self.pos[rank][self.pref[p]]
                        for p in range(self.pref_qty)
                    ])
                ) #positions(descending) corresponding to t items  
            
                # Obtain the resulting order from the optimization map
                for r in ROWS:
                    for c in COLS:
                        if value(solution_vars[r][c]) > 0.99: #put item r in position c
                            '''
                            result[r] = c
                            result_pos[r] = sorted_pos[c]
                            '''
                            result[r] = sorted_pos[c]
                            result_pos[r] = c


                # Update vars for next iteration
            
                for p in range(self.pref_qty):
                    subj = self.pref[p]
                    self.pos[rank][subj] = result_pos[p]
                    self.acumm_A[subj] += self.__pos_bias(p+1)
                    self.acumm_R[subj] += self.rank_series[rank][subj]

            # Save this iteration's unfairness values
            self.unfairness_vals.append(
                sum([
                    l1_norm_dist(self.acumm_A[s], self.acumm_R[s])
                    for s in range(self.subj_qty)
                ])
            )

            print("OK")
        
        # output final rankings
        for u in range(self.rank_qty):
            self.final[u] = argsort(self.pos[u])
            
            tem_list = []
            for j in self.final[u]:
                tem_list.append(self.rank_series[u][j])

            self.rerank_rel[u] = tem_list
            
            

    def get_result(
        self: "ILPBased",
    ):
        """Returns unfairness values and acummulated variables
        """
        return {
            "unfairness_vals": self.unfairness_vals,
            "acummulated_attention": self.acumm_A,
            "acummulated_relevance": self.acumm_R,
        }
