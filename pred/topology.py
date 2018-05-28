# coding:utf-8
# Created by chen on 26/05/2018
# email: q.chen@student.utwente.nl

# import igraph
from igraph import *
import pandas
import numpy as np
from scipy.sparse import csgraph
import itertools

class topology(object):
    def __init__(self, graph):
        self.graph = graph
        self.outdegrees = graph.outdegree()
        self.indegrees = graph.indegree()
        self.adjlistOut = list(map(set, graph.get_adjlist()))
        self.adjlistIn = list(map(set, graph.get_adjlist(mode=IN)))
        self.adjmatrix = np.matrix([graph.get_adjacency()[i] for i in range(len(graph.vs))])
        self.m = graph.ecount()  # number of edges in the graph
        self.n = graph.vcount()  # vertice number

    '''
    social theory based metric
    '''

    def common_neighbors(self, i, j, mode="out"):
        if mode == "out":
            inters = self.adjlistOut[i].intersection(self.adjlistOut[j])
            if inters:
                return inters
            else:
                return []
        elif mode == "in":
            inters = self.adjlistIn[i].intersection(self.adjlistIn[j])
            if inters:
                return inters
            else:
                return []

    def degree_cenrtality(self, mode="out"):
        if mode == "out":
            denominator = self.n - 1
            if denominator:
                return np.array(self.outdegrees) / (self.n - 1)
            else:
                return 0
        elif mode == "in":
            denominator = self.n - 1
            if denominator:
                return np.array(self.indegrees) / (self.n - 1)
            else:
                return 0

    def closeness_centrality(self, mode="out"):
        if mode == "out":
            return self.graph.closeness(mode=OUT)
        elif mode == "in":
            return self.graph.closeness(mode=IN)

    # directed acyclic graohs will have an all-zero eigenvector centrality vector
    # directed graphs with negative weights may have eigenvector centrality vectors with complex members
    # if a vertex is not in a strongly connected component of size at least 2, or in the out-componnet of such a compoent,
    # then its eigenvector centrality will be zero
    def eigen_cen(self):
        return self.graph.evcent()
        # return self.graph.eigenvector_centrality()

    def betweenness_centrality(self):
        return self.graph.betweenness()

    def pageRank_centrality(self):
        return self.pageRank_centrality()

    # a model based on the node centrality and weak-tie theory for link-prediction
    # the signifiance of each common neighbor of two nodes is different according to their centralities
    # beta >1: it will amplify the contribution, otherwise, it can restrain the contribution
    def weak_tie_based(self, cent="betweenness", beta=-1, u=None, v=None):
        if cent == "betweenness":
            cen = self.betweenness_centrality()
        elif cent == "closness":
            cen = self.closeness_centrality()
        elif cent == "degree":
            cen = self.degree_cenrtality()
        elif cent == "eigen":
            cen = self.eigen_cen()
        elif cent == "pageRank":
            cen = self.pageRank_centrality()
        if u == None and v == None:
            simScore = np.zeros([self.n, self.n])
            for i in range(self.n):
                for j in range(self.n):
                    commonNeighs = self.common_neighbors(i, j)
                    if commonNeighs:
                        simScore[i][j] = sum([np.power(cen[x], beta) for x in commonNeighs if cen[x]])
            return simScore
        else:
            commonNeighs = self.common_neighbors(u, v)
            return sum([np.power(cen[x], beta) for x in commonNeighs if cen[x]])

    '''
    random walk based metric
    '''

    def laplacian_matrix(self):
        # return csgraph.laplacian(self.adjmatrix, normed=False,return_diag=True,use_out_degree=True)
        return csgraph.laplacian(self.adjmatrix, normed=False, use_out_degree=True)

    def pse_laplacian(self):
        return np.linalg.pinv(self.laplacian_matrix())

    # hitting time only for undirectd and unweightd graph
    # commute time: since the hitting time metric is not symmetric
    # commute time count the expected steps both from x to y and from y to x
    def CT(self, i, j):
        pseLaplacian = self.pse_laplacian()
        return self.m * (pseLaplacian[i, i] + pseLaplacian[j, j] - 2 * pseLaplacian[i, j])

    # cosine similarity time (CST): calcualte the similarity of two vectors
    def CST(self, i, j):
        pseLaplacian = self.pse_laplacian()
        denominator = pseLaplacian[i, i] * pseLaplacian[j, j]
        if denominator > 0:
            return pseLaplacian[i, j] / np.sqrt(denominator)
        else:
            return 0

    # SimRank: SimRank is defined in a self-constraint way
    # according to the asssumption that two nodes are similar if they are connected to similar nods
    # calculate simRank based on  outdegree
    # regular simRank is based on indegree, return the whole matrix
    def simRank(self, r=0.8, max_iter=100, eps=1e-4, mode="out"):
        sim_prev = np.zeros(self.n)
        sim = np.identity(self.n)
        for i in range(max_iter):
            if np.allclose(sim, sim_prev, atol=eps):  # iterate until converge
                break
            sim_prev = np.copy(sim)
            if mode == "out":
                for u in range(self.n):
                    for v in range(self.n):
                        if u == v:  # if u and v is equal return 1
                            continue
                        if self.outdegrees[u] == 0 or self.outdegrees[v] == 0:
                            sim[u][v] = 0
                        else:
                            s_uv = sum([sim_prev[neighbor_u][neighbor_v] for neighbor_u, neighbor_v in
                                        itertools.product(self.adjlistOut[u], self.adjlistOut[v])])
                            sim[u][v] = (r * s_uv) / (self.outdegrees[u] * self.outdegrees[v])
            elif mode == "in":
                for u in range(self.n):
                    for v in range(self.n):
                        if u == v:  # if u and v is equal return 1
                            continue
                        if self.indegrees[u] == 0 or self.indegrees[v] == 0:
                            sim[u][v] = 0
                        else:
                            s_uv = sum([sim_prev[neighbor_u][neighbor_v] for neighbor_u, neighbor_v in
                                        itertools.product(self.adjlistIn[u], self.adjlistIn[v])])
                            sim[u][v] = (r * s_uv) / (self.indegrees[u] * self.indegrees[v])
        return sim

    # rooted PageRank (RPR): Rooted PageRank is a modification of PageRank
    def RPR(self, factor=0.5):
        D_inv = np.diag(1. / np.array([max(x, 1) for x in self.outdegrees]))
        try:
            return (1 - factor) * (np.linalg.inv(np.identity(self.n) - D_inv.dot(self.adjmatrix)))
        except:
            return np.zeros([self.n, self.n])

    '''
    neighbor based metric
    '''

    def degree_product(self, i, j, mode="out"):
        if mode == "out":
            return self.outdegrees[i] * self.outdegrees[j]
        elif mode == "in":
            return self.indegrees[i] * self.indegrees[j]

    def all_neighbors(self, i, j, mode="out"):
        if mode == "out":
            return self.adjlistOut[i].union(self.adjlistOut[j])
        elif mode == "in":
            return self.adjlistIn[i].union(self.adjlistIn[j])

    # CN: common neighborhoods
    def CN(self, i, j, mode="out"):
        if mode == "out":
            return len(self.common_neighbors(i, j))
        elif mode == "in":
            return len(self.common_neighbors(i, j, mode="in"))

    # Jaccard_coeff: normalize the size of common neighbors
    def JC(self, i, j, mode="öut"):
        if mode == "out":
            denominator = len(self.all_neighbors(i, j))
            if denominator:
                return 1.0 * len(self.common_neighbors(i, j)) / denominator
            else:
                return 0
        elif mode == "in":
            denominator = len(self.all_neighbors(i, j, mode="in"))
            if denominator:
                return 1.0 * len(self.common_neighbors(i, j, mode="in")) / denominator
            else:
                return 0

    # soresen_index: besides considering the size of the common neighbors, it also points out lower degrees of nodes wouldj have higher link likelihood
    def SI(self, i, j, mode="out"):
        if mode == "out":
            denometor = (self.outdegrees[i] + self.outdegrees[j])
            if denometor:
                return 1.0 * len(self.common_neighbors(i, j)) / denometor
            else:
                return 0
        elif mode == "in":
            denometor = (self.indegrees[i] + self.indegrees[j])
            if denometor:
                return 1.0 * len(self.common_neighbors(i, j, mode="in")) / denometor
            else:
                return 0

    # salton_cosin_sim: cosine matric
    def SC(self, i, j, mode="out"):
        if mode == "out":
            denominator = self.outdegrees[i] * self.outdegrees[j]
            if denominator > 0:
                return 1.0 * len(self.common_neighbors(i, j)) / np.sqrt(denominator)
            else:
                return 0
        elif mode == "in":
            denominator = self.indegrees[i] * self.indegrees[j]
            if denominator > 0:
                return 1.0 * len(self.common_neighbors(i, j, mode="in")) / np.sqrt(denominator)
            else:
                return 0

    # Hub_Promoted: HP defines the topological overlap of nodex and y
    # the HP value is deterined by the lower degree of nodes
    def HP(self, i, j, mode="out"):
        if mode == "out":
            denominator = np.minimum(self.outdegrees[i], self.outdegrees[j])
            if denominator:
                return 1.0 * len(self.common_neighbors(i, j)) / denominator
            else:
                return 0
        elif mode == "in":
            denominator = np.minimum(self.indegrees[i], self.indegrees[j])
            if denominator:
                return 1.0 * len(self.common_neighbors(i, j, mode="in")) / denominator
            else:
                return 0

    # Hub_depressed: hub depressed, similar to HP, but the value is determined by the higher degree of nodes
    def HD(self, i, j, mode="out"):
        if mode == "our":
            denomiantor = np.maximum(self.outdegrees[i], self.outdegrees[j])
            if denomiantor:
                return 1.0 * len(self.common_neighbors(i, j)) / denomiantor
            else:
                return 0
        elif mode == "in":
            denomiantor = np.maximum(self.indegrees[i], self.indegrees[j])
            if denomiantor:
                return 1.0 * len(self.common_neighbors(i, j, mode="in")) / denomiantor
            else:
                return 0

    # Leicht_holem_Nerman: assign high similarity to node pairs that have many common neighbors compared not to the possible maximum
    # but to the expected number of such neighbors
    def LHN(self, i, j, mode="out"):
        if mode == "out":
            denominator = (self.outdegrees[i] * self.outdegrees[j])
            if denominator:
                return 1.0 * len(self.common_neighbors(i, j)) / denominator
            else:
                return 0
        elif mode == "in":
            denominator = (self.indegrees[i] * self.indegrees[j])
            if denominator:
                return 1.0 * len(self.common_neighbors(i, j, mode="in")) / denominator
            else:
                return 0

    # Parameter_dependent: lamda is a free parameter, when lambda = 0, PD degenerates to CN
    # if lambda = 0.1 and lambda =1, it degenerates to Salton and LHN metric, respectively
    def PD(self, i, j, lam=0.5, mode="out"):
        if mode == "out":
            denominator = self.outdegrees[i] * self.outdegrees[j]
            if denominator:
                return 1.0 * len(self.common_neighbors(i, j)) / np.power(denominator, lam)
            else:
                return 0
        elif mode == "in":
            denominator = self.indegrees[i] * self.indegrees[j]
            if denominator:
                return 1.0 * len(self.common_neighbors(i, j, mode="in")) / np.power(denominator,
                                                                                    lam)
            else:
                return 0

    # AA: widely used in social network
    # the AA measure is formulated related to Jaccard's coefficent
    # but here, common neighbos which have fewer neighbors are weighted more heavily
    def AA(self, i, j, mode="out"):
        if mode == "out":
            common_neighbors = self.common_neighbors(i, j)
            return sum([1 / np.log(self.outdegrees[item]) for item in common_neighbors if
                        self.outdegrees[item] and self.outdegrees[item] != 1])
        elif mode == "in":
            common_neighbors = self.common_neighbors(i, j, mode="in")
            return sum([1 / np.log(self.indegrees[item]) for item in common_neighbors if
                        self.indegrees[item] and self.indegrees[item] != 1])

    def AA_matrix(self, mode="out"):
        simMatrix = np.zeros([self.n, self.n])
        if mode == "out":
            for i in range(self.n):
                for j in range(self.n):
                    common_neighbors = self.common_neighbors(i, j)
                    outdegreelists = [self.outdegrees[item] for item in common_neighbors if
                                      self.outdegrees[item] and self.outdegrees[item] != 1]
                    if outdegreelists:
                        simMatrix[i][j] = sum(
                            [1 / np.log(item) for item in outdegreelists]
                        )
                    else:
                        simMatrix[i][j] = 0
        elif mode == "in":
            for i in range(self.n):
                for j in range(self.n):
                    common_neighbors = self.common_neighbors(i, j, mode="in")
                    indegreelists = [self.indegrees[item] for item in common_neighbors if
                                     self.indegrees[item] and self.indegrees[item] != 1]
                    if indegreelists:
                        simMatrix[i][j] = sum(
                            [1 / np.log(item) for item in indegreelists])
                    else:
                        simMatrix[i][j] = 0

        return simMatrix

    # preferential_attachemnt: preferential attachement:
    # the PA metric indeicates that new lnks will be more likely to connect high-degree nodes than lower ones
    def PA(self, i, j, mode="out"):
        if mode == "öut":
            return self.outdegrees[i] * self.outdegrees[j]
        elif mode == "in":
            return self.indegrees[i] * self.indegrees[j]

    # Resource_Allocation: motivated by the physical processes of resource allocatiojn
    # RA metric has a similar form like AA
    #     RA metric punishes the high-degree common neighbos more heavily tahn A
    # RA performs better for the network with high average degrees
    def RA(self, i, j, mode="out"):
        if mode == "out":
            common_neighbors = self.common_neighbors(i, j)
            degreeslist = [self.outdegrees[item] for item in common_neighbors]
            if common_neighbors and all(v == 0 for v in degreeslist):
                return sum([1 / self.outdegrees[item] for item in common_neighbors if self.outdegrees[item]])
            else:
                return 0
        elif mode == "in":
            common_neighbors = self.common_neighbors(i, j, mode="in")
            degreeslist = [self.indegrees[item] for item in common_neighbors]
            if common_neighbors and all(v == 0 for v in degreeslist):
                return sum([1 / self.indegrees[item] for item in common_neighbors if self.indegrees[item]])
            else:
                return 0

    '''
    path based metric
    '''

    # local path: make use of informtaion of local paths with length 2 and length 3
    def LP(self, alpha):
        return np.linalg.matrix_power(self.adjmatrix, 2) + alpha * (np.linalg.matrix_power(self.adjmatrix, 3))

    # Katz: katz metirc is based on ensemble of all paths
    # the paths are exponential dampled by length that can give more weiggts to the shorter paths
    # one can verify that the matrix of scores is given by (inverse of(I- beta*M))-I
    def katz(self, beta):
        return np.linalg.inv(np.identity(self.n) - beta * self.adjmatrix) - np.identity(self.n)

    # FindLink: reference paper: Fast and accurate link prediction in social networking systems
    # L is teh maximum length of paths explored in G
    # todo: fill the score
    def FL(self, L):
        # adjMatrix = self.adjmatrix
        adjMatrix = np.zeros((self.n, self.n), dtype=object)
        for i in range(self.n):
            for j in range(self.n):
                if self.adjmatrix[i, j] == 1:
                    adjMatrix[i, j] = str(j)
                else:
                    adjMatrix[i, j] = 0
        for count in range(2, L + 1):
            for i in range(self.n):
                for j in range(self.n):
                    for k in range(self.n):
                        if adjMatrix[i, k] and adjMatrix[k, j]:
                            adjMatrix[i, j] = str(adjMatrix[i, k]) + "," + str(adjMatrix[k, j])
            # for i in range(self.n):
            #     for j in range(self.n):
            #         denominator = 1
            #         for k in range(1,i):
            #             denominator = denominator*(self.n - k)
            #     simScore[i,j] = simScore[i,j]+(1/(i-1))*
        return adjMatrix
