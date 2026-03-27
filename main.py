import numpy as np
from scipy.spatial.distance import pdist, squareform
from numpy.random import gamma, poisson
import random
from typing import Optional, Union
from matplotlib import pyplot as plt


def random_graph(num_nodes: int, maxx: float = 10) -> np.array:
    """
    Randomly generates points in 2D and returns matrix of pairwise euclidean distances.
    Thus underlying graph of matrix is fully connected and satisfies triangle inequality.

    :param num_nodes:
    :return:
    """
    points = np.random.rand(num_nodes, 2)*maxx  # 2d points uniformly distributed in [0, maxx)
    print(points)
    euclids = pdist(points)  # calculates pairwise distances of each point
    return squareform(euclids)


# def dist(graph, i, j, m):
#     return graph[m * i + j - ((i + 2) * (i + 1)) // 2]

def greedy_solver(dist: np.array, reward: np.array, start: int, time_window: float, epsilon: float = .0) -> tuple[list[int], int]:
    """
    Approach:
    - calculate shortest path from `start` node to all other nodes
        - we know for each node d(node, start): the time to return to start
        - we already have this because graph is fully connected and satisfies triangle inequality
    - define the following ratio heuristic while traversing:
        - score(v) = reward(v)/(d(node, v) + d(node, finish))

    Now from starting node always pick node with highest ratio where d(node, v) + d(node, finish) <= time_window.

    :param dist:
    :param reward:
    :param time_window:
    :return: route and return of route
    """
    # verify we have an adjacency matrix
    assert dist.ndim == 2
    assert dist.shape[0] == dist.shape[1]

    num_nodes = dist.shape[0]

    def score(neighbor: int, predecessor: int) -> float:
        # ratio heuristic
        return reward[neighbor]/(dist[predecessor, neighbor] + dist[neighbor, start])

    def pick_best(v: int, unvisited: set[int]) -> int:
        # better even would be simulated annealing
        if np.random.rand() < epsilon:
            return random.choice(list(unvisited))

        # picks best unvisited node given the current node and remaining time, otherwise returns start.
        best_score = float("-inf")
        best_neighbor = -1

        for neighbor in unvisited:
            # check if this neighbor can still reach finish in time
            rem_time = time_window - total_time
            if dist[v, neighbor] + dist[neighbor, start] > rem_time:
                continue

            score_n = score(neighbor, v)
            if score_n > best_score:
                best_score = score_n
                best_neighbor = neighbor

        # if no neighbor found, then just return to start
        if best_neighbor == -1:
            return start

        return best_neighbor

    unvisited = set([i for i in range(num_nodes)]) - {start}
    total_time = 0
    current = pick_best(start, unvisited)
    route = [start, current]
    prev = start
    ret = reward[start]

    # we are just repeatedly picking best neighbor and updating relevant values
    while current != start and total_time <= time_window:
        unvisited.remove(current)
        ret += reward[current]
        total_time += dist[prev, current]
        prev = current
        current = pick_best(current, unvisited)
        route.append(current)

    return route, ret


def main(dist: np.array, node_params: np.array, time_window: float, num_days: int, epsilon: float):
    """
    Find optimal route in a graph that satisfies time constraint.
    Route is optimal if its expected return is highest wrt all other feasible routes.
    Return of route is the sum of node rewards.
    The node reward is a value sampled from a node-specific distribution after node visited.
    The time constraint is satisfied if the path cost (sum of edge costs) is leq than time_window.

    This problem combines the orienteering problem - finding path that maximizes rewards and
    time constraint (NP-hard) - with the exploration-/exploitation problem as we don't know
    node reward distributions (MAB problem).

    (Can be viewed as MAB where actions are all possible paths)

    "Practical" example: You are a pickpocket and try to find a route through the city which allows you to steal from
    as many people as possible in the spots along the route. Some spots are more lucrative as maybe they are more
    crowded and thus have higher reward.

    Note that another variant of this problem might model a bust. I.e. you are caught and have to pay a fine. A bust
    would also end the route.

    :param dist: adjacency matrix of graph
    :param node_params: true parameters of node distributions (using poisson distribution => just lambdas)
    :param time_window: max time of a route
    :param num_days: total number of routes we are allowed to walk
    :return:
    """
    num_nodes = dist.shape[0]
    route_days = []
    params_days = []

    # initialize distribution params
    # we assume node rewards are poisson distributed, thus posterior of param (lambda) is gamma distributed
    est_post_params = [(1, 1) for _ in range(num_nodes)]  # initial alpha, beta for Gamma(alpha, beta)

    # main loop:
    for day in range(num_days):
        # thompson sample node EVs based on current param posteriors
        thomps_rewards = [gamma(alpha, scale=1/beta) for alpha, beta in est_post_params]

        # run orienteering solver
        route, ret = greedy_solver(dist, thomps_rewards, start=0, time_window=time_window, epsilon=epsilon)
        route_days.append(route)

        # sample and update
        for node in route:
            # sample rewards from true distribution of nodes visited in the route
            reward_sample = poisson(lam=node_params[node])
            # update posterior params given new samples
            alpha, beta = est_post_params[node]
            alpha += reward_sample
            beta += 1
            est_post_params[node] = (alpha, beta)

        params_days.append(est_post_params)

    return est_post_params, route_days, params_days


class RouteLearner:
    """
    Essentially the same as above but easier to track intermediate values, e.g. for visualization.
    """

    def __init__(self, points: np.array, true_lambdas: np.array, time_window: float):
        """

        :param points: num_points x dim array
        :param true_lambdas:
        :param time_window:
        """
        self.points = points
        self.num_points = points.shape[0]
        self.dist = squareform(pdist(points))  # calculates pairwise distances of each point
        self.true_lambdas: np.array = np.array(true_lambdas) if isinstance(true_lambdas, list) else true_lambdas
        self.lambs_posterior_params: list[tuple[float, float]] = []

        self.time_window = time_window
        self.start = 0

    def init_priors(self, priors: Optional[Union[tuple[float, float], list[tuple[float, float]]]] = None) -> None:
        if priors is not None:
            if isinstance(priors, tuple):
                assert len(priors) == 2
                self.lambs_posterior_params = [priors for _ in range(self.num_points)]
            elif isinstance(priors, list):
                assert all(len(p) == 2 for p in priors)
                self.lambs_posterior_params = [p for p in priors]
            else:
                raise ValueError("Priors must be tuple of length 2 or list of tuples of length 2.")
        else:
            self.lambs_posterior_params = [(1, 1) for _ in range(self.num_points)]

    def thompson_sample(self, epsilon) -> list[tuple[int, float]]:
        """
        Does a thompson sample:
        - samples rewards based on current posterior
        - applies greedy solver to get route
        - sample rewards of nodes in route
        - return nodes and samples

        In essence we sample a route and a return (sort of like a MAB sample).
        """

        if len(self.lambs_posterior_params) != self.num_points:
            raise ValueError("Priors not correctly initialized. Call init_priors prior to this method.")

        # thompson sample node EVs based on current param posteriors
        thomps_rewards = [gamma(alpha, scale=1/beta) for alpha, beta in self.lambs_posterior_params]

        # run orienteering solver
        route, ret = greedy_solver(
            self.dist, thomps_rewards, start=self.start, time_window=self.time_window, epsilon=epsilon
        )

        # sample and save
        route_info = []
        for node in route:
            # sample rewards from true distribution of nodes visited in the route
            reward_sample = poisson(lam=self.true_lambdas[node])
            route_info.append((node, reward_sample))

        return route_info

    def thompson_update(self, route_info: list[tuple[int, float]]) -> None:
        """Performs a thompson update of the posteriors based on sampled rewards of nodes in the route"""
        for node, reward_sample in route_info:
            # update posterior params given the reward samples
            alpha, beta = self.lambs_posterior_params[node]
            alpha += reward_sample
            beta += 1
            self.lambs_posterior_params[node] = (alpha, beta)

    def ev_route(self):
        return greedy_solver(self.dist, self.true_lambdas, start=self.start, time_window=self.time_window)

    def ev_estimate(self) -> np.array:
        """Returns the current estimates of the lambdas of each node"""
        return np.array([alpha / beta for alpha, beta in self.lambs_posterior_params])


if __name__ == "__main__":
    np.random.seed(9582735)
    random.seed(43875634)

    num_nodes = 20
    dist = random_graph(num_nodes)
    time_window = 15
    num_days = 200
    true_lambdas = np.random.lognormal(mean=2, sigma=1, size=num_nodes)
    print(f"True lambdas: {true_lambdas.tolist()}")

    gamma_params, route_days, params_days = main(dist, true_lambdas, time_window, num_days, epsilon=0.05)

    est_lambs = [alpha/beta for alpha, beta in gamma_params]
    print(f"Estimated lambdas: {est_lambs}")
    optimal_route, opt_ret = greedy_solver(dist, true_lambdas, start=0, time_window=time_window)
    print()
    # print(optimal_route, opt_ret)
    est_lambs_days = np.array([[alpha/beta for alpha, beta in day_param] for day_param in params_days])
    print(np.mean(true_lambdas[None,] - est_lambs_days))
    print(route_days[-5:])
    print()

    returns = np.array([sum([true_lambdas[node] for node in route]) for route in route_days])
    final_route, final_ret = greedy_solver(dist, est_lambs_days[-1], start=0, time_window=time_window)
    print(opt_ret, final_ret)
    regret = opt_ret - returns
