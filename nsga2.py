import numpy as np
import os

def find_the_better(x, y):
    if isinstance(x, list):
        x = np.array(x)
    if isinstance(y, list):
        y = np.array(y)
    sub_ = x - y
    x_better = np.all(sub_ <= 0)
    y_better = np.all(sub_ >= 0)
    if x_better == y_better:  # True - True
        return -1
    if y_better:  # False - True
        return 1
    return 0  # True - False

class ElitistArchive:
    def __init__(self):
        self.id, self.F = [], []

    def update(self, id, fitness):
        F = np.array(fitness)

        if id not in self.id:
            l = len(self.F)
            r = np.zeros(l, dtype=np.int8)
            status = True
            for i, F_ in enumerate(self.F):
                better_idv = find_the_better(F, F_)
                if better_idv == 0:
                    r[i] += 1
                elif better_idv == 1:
                    status = False
                    break
            if status:
                self.id.append(id)
                self.F.append(F)
                r = np.append(r, 0)

                self.id = np.array(self.id)[r == 0].tolist()
                self.F = np.array(self.F)[r == 0].tolist()

    def save(self, save_fol, itr, verbose=True):
        os.makedirs(save_fol, exist_ok=True)
        with open(save_fol + f"/itr-{itr}.txt", "w") as f:
            f.writelines(str(self.id) + "\n")
            f.writelines(str(self.F))

        if verbose:
            print("Saved current Pareto Front")



from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.util.randomized_argsort import randomized_argsort
import json


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

class RankAndCrowdingSurvival:
    def __init__(self):
        self.name = 'Rank and Crowding Survival'

    @staticmethod
    def do(F, n_survive):
        # get the objective space values and objects
        pop = np.arange(len(F))
        F = np.array(F)
        
        # the final indices of surviving individuals
        survivors = []

        # do the non-dominated sorting until splitting front
        fronts = NonDominatedSorting().do(F, n_stop_if_ranked=n_survive)

        for k, front in enumerate(fronts):

            # calculate the crowding distance of the front
            crowding_of_front = calculating_crowding_distance(F[front, :])

            # current front sorted by crowding distance if splitting
            if len(survivors) + len(front) > n_survive:
                I = randomized_argsort(crowding_of_front, order='descending', method='numpy')
                I = I[:(n_survive - len(survivors))]

            # otherwise take the whole front unsorted
            else:
                I = np.arange(len(front))

            # extend the survivors by all or selected individuals
            survivors.extend(front[I])
        return pop[survivors]


def calculating_crowding_distance(F):
    infinity = 1e+14

    n_points = F.shape[0]
    n_obj = F.shape[1]

    if n_points <= 2:
        return np.full(n_points, infinity)
    else:

        # sort each column and get index
        I = np.argsort(F, axis=0, kind='mergesort')

        # now really sort the whole array
        F = F[I, np.arange(n_obj)]

        # get the distance to the last element in sorted list and replace zeros with actual values
        dist = np.concatenate([F, np.full((1, n_obj), np.inf)]) - np.concatenate([np.full((1, n_obj), -np.inf), F])

        index_dist_is_zero = np.where(dist == 0)

        dist_to_last = np.copy(dist)
        for i, j in zip(*index_dist_is_zero):
            dist_to_last[i, j] = dist_to_last[i - 1, j]

        dist_to_next = np.copy(dist)
        for i, j in reversed(list(zip(*index_dist_is_zero))):
            dist_to_next[i, j] = dist_to_next[i + 1, j]

        # normalize all the distances
        norm = np.max(F, axis=0) - np.min(F, axis=0)
        norm[norm == 0] = np.nan
        dist_to_last, dist_to_next = dist_to_last[:-1] / norm, dist_to_next[1:] / norm

        # if we divided by zero because all values in one columns are equal replace by none
        dist_to_last[np.isnan(dist_to_last)] = 0.0
        dist_to_next[np.isnan(dist_to_next)] = 0.0

        # sum up the distance to next and last and norm by objectives - also reorder from sorted list
        J = np.argsort(I, axis=0)
        crowding = np.sum(dist_to_last[J, np.arange(n_obj)] + dist_to_next[J, np.arange(n_obj)], axis=1) / n_obj

    # replace infinity with a large number
    crowding[np.isinf(crowding)] = infinity
    return crowding