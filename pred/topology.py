# coding:utf-8
# Created by chen on 26/05/2018
# email: q.chen@student.utwente.nl

# import igraph
from igraph import *
import pandas
import numpy as np
from scipy.sparse import csgraph
import itertools

nodesListData = pandas.read_csv("XS_node.csv")
reassignId =nodesListData["id"].tolist()
getDict = {elem:count for count,elem in enumerate(reassignId)}
# print(getDict)
# nodesListData = nodesListData.sort_values(by=['id'])
edgeListData = pandas.read_csv("XS_edge.csv")

edgeListData["from"].replace(getDict, inplace=True)
edgeListData["to"].replace(getDict, inplace=True)
subset = edgeListData[["from", "to"]]
edgeList = [tuple(x) for x in subset.values]

vertices = nodesListData["user"].tolist()
urlList = nodesListData["url"].tolist()
g = Graph(vertex_attrs={"label": vertices, "url": urlList}, edges=edgeList, directed=True)

g = Graph.simplify(g)
#  remove isolated nodes
g.delete_vertices(g.vs.select(_degree=0))

# visual_style = {}
# # Scale vertices based on degree
# outdegree = g.outdegree()
# visual_style["vertex_size"] = [x / max(outdegree) * 25 + 50 for x in outdegree]
#
# # Set bbox and margin
# visual_style["bbox"] = (800, 800)
# visual_style["margin"] = 100
#
# # Define colors used for outdegree visualization
# colours = ['#fecc5c', '#a31a1c']
#
# # Order vertices in bins based on outdegree
# bins = np.linspace(0, max(outdegree), len(colours))
# digitized_degrees = np.digitize(outdegree, bins)
#
# # Set colors according to bins
# g.vs["color"] = [colours[x - 1] for x in digitized_degrees]
#
# # Also color the edges
# for ind, color in enumerate(g.vs["color"]):
#     edges = g.es.select(_source=ind)
#     edges["color"] = [color]
#
# # Don't curve the edges
# visual_style["edge_curved"] = False
#
# # Community detection
# communities = g.community_edge_betweenness(directed=True)
# clusters = communities.as_clustering()
#
# # Set edge weights based on communities
# weights = {v: len(c) for c in clusters for v in c}
# g.es["weight"] = [weights[e.tuple[0]] + weights[e.tuple[1]] for e in g.es]
#
# # Choose the layout
# N = len(vertices)
# visual_style["layout"] = g.layout_fruchterman_reingold(weights=g.es["weight"], maxiter=1000, area=N ** 3,
#                                                        repulserad=N ** 3)
# # Plot the graph
# plot(g, **visual_style)


class topology(object):
    def __init__(self, graph):
        self.graph = graph
        self.outdegrees = graph.outdegree()
        self.indegrees = graph.indegree()
        self.adjlistOut = list(map(set,graph.get_adjlist()))
        self.adjlistIn = list(map(set,graph.get_adjlist(mode=IN)))
        self.adjmatrix = np.matrix([graph.get_adjacency()[i] for i in range(len(graph.vs))])
        self.m = graph.ecount() #number of edges in the graph
        self.n = graph.vcount() # vertice number


    '''
    social theory based metric
    '''
    def common_neighbors(self, i, j,mode="out"):
        if mode == "out":
            return self.adjlistOut[i].intersection(self.adjlistOut[j])
        elif mode == "in":
            return self.adjlistIn[i].intersection(self.adjlistIn[j])

    def degree_cenrtality(self,mode="out"):
        if mode == "out":
            return np.array(self.outdegrees) / (self.n-1)
        elif mode == "in":
            return np.array(self.indegrees) / (self.n-1)

    def closeness_centrality(self,mode="out"):
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
    def weak_tie_based(self,cent = "betweenness",beta=-1):
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

        simScore = np.zeros([self.n,self.n])
        for i in range(self.n):
            for j in range(self.n):
                commonNeighs = prede.common_neighbors(i,j)
                if commonNeighs:
                        simScore[i][j] = sum([np.power(cen[x],beta) for x in commonNeighs if cen[x]])
        return simScore

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
        return pseLaplacian[i, j] / np.sqrt(pseLaplacian[i, i] * pseLaplacian[j, j])

    # SimRank: SimRank is defined in a self-constraint way
    # according to the asssumption that two nodes are similar if they are connected to similar nods
    # calculate simRank based on  outdegree
    # regular simRank is based on indegree
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
        return (1 - factor) * (np.linalg.inv(np.identity(self.n) - D_inv.dot(self.adjmatrix)))

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
            return 1.0 * len(self.common_neighbors(i, j)) / len(self.all_neighbors(i, j))
        elif mode == "in":
            return 1.0 * len(self.common_neighbors(i, j, mode="in")) / len(self.all_neighbors(i, j, mode="in"))

    # soresen_index: besides considering the size of the common neighbors, it also points out lower degrees of nodes wouldj have higher link likelihood
    def SI(self, i, j, mode="out"):
        if mode == "out":
            return 1.0 * len(self.common_neighbors(i, j)) / (self.outdegrees[i] + self.outdegrees[j])
        elif mode == "in":
            return 1.0 * len(self.common_neighbors(i, j, mode="in")) / (self.indegrees[i] + self.indegrees[j])

    # salton_cosin_sim: cosine matric
    def SC(self, i, j, mode="out"):
        if mode == "out":
            return 1.0 * len(self.common_neighbors(i, j)) / np.sqrt(self.outdegrees[i] * self.outdegrees[j])
        elif mode == "in":
            return 1.0 * len(self.common_neighbors(i, j, mode="in")) / np.sqrt(self.indegrees[i] * self.indegrees[j])

    # Hub_Promoted: HP defines the topological overlap of nodex and y
    # the HP value is deterined by the lower degree of nodes
    def HP(self, i, j, mode="out"):
        if mode == "out":
            return 1.0 * len(self.common_neighbors(i, j)) / np.minimum(self.outdegrees[i], self.outdegrees[j])
        elif mode == "in":
            return 1.0 * len(self.common_neighbors(i, j, mode="in")) / np.minimum(self.indegrees[i], self.indegrees[j])

    # Hub_depressed: hub depressed, similar to HP, but the value is determined by the higher degree of nodes
    def HD(self, i, j, mode="out"):
        if mode == "our":
            return 1.0 * len(self.common_neighbors(i, j)) / np.maximum(self.outdegrees[i], self.outdegrees[j])
        elif mode == "in":
            return 1.0 * len(self.common_neighbors(i, j, mode="in")) / np.maximum(self.indegrees[i], self.indegrees[j])

    # Leicht_holem_Nerman: assign high similarity to node pairs that have many common neighbors compared not to the possible maximum
    # but to the expected number of such neighbors
    def LHN(self, i, j, mode="out"):
        if mode == "out":
            return 1.0 * len(self.common_neighbors(i, j)) / (self.outdegrees[i] * self.outdegrees[j])
        elif mode == "in":
            return 1.0 * len(self.common_neighbors(i, j, mode="in")) / (self.indegrees[i] * self.indegrees[j])

    # Parameter_dependent: lamda is a free parameter, when lambda = 0, PD degenerates to CN
    # if lambda = 0.1 and lambda =1, it degenerates to Salton and LHN metric, respectively
    def PD(self, i, j, lam, mode="out"):
        if mode == "out":
            return 1.0 * len(self.common_neighbors(i, j)) / np.power(self.outdegrees[i] * self.outdegrees[j], lam)
        elif mode == "in":
            return 1.0 * len(self.common_neighbors(i, j, mode="in")) / np.power(self.indegrees[i] * self.indegrees[j],
                                                                                lam)

    # AA: widely used in social network
    # the AA measure is formulated related to Jaccard's coefficent
    # but here, common neighbos which have fewer neighbors are weighted more heavily
    def AA(self, i, j, mode="out"):
        if mode == "out":
            common_neighbors = self.common_neighbors(i, j)
            return sum([1 / np.log(self.outdegrees[item]) for item in common_neighbors if self.outdegrees[item]])
        elif mode == "in":
            common_neighbors = self.common_neighbors(i, j, mode="in")
            return sum([1 / np.log(self.indegrees[item]) for item in common_neighbors if self.indegrees[item]])

    def AA_matrix(self, mode="out"):
        simMatrix = np.zeros([self.n, self.n])
        if mode == "out":
            for i in range(self.n):
                for j in range(self.n):
                    common_neighbors = self.common_neighbors(i, j)
                    simMatrix[i][j] = sum(
                        [1 / np.log(self.outdegrees[item]) for item in common_neighbors if self.outdegrees[item]])
        elif mode == "in":
            for i in range(self.n):
                for j in range(self.n):
                    common_neighbors = self.common_neighbors(i, j, mode="in")
                    simMatrix[i][j] = sum(
                        [1 / np.log(self.indegrees[item]) for item in common_neighbors if self.indegrees[item]])

        return simMatrix

    # preferential_attachemnt: preferential attachement:
    # the PA metric indeicates that new lnks will be more likely to connect high-degree nodes than lower ones
    def PA(self, i, j,mode="out"):
        if mode == "öut":
            return self.outdegrees[i] * self.outdegrees[j]
        elif mode =="in":
            return self.indegrees[i] * self.indegrees[j]

    # Resource_Allocation: motivated by the physical processes of resource allocatiojn
    # RA metric has a similar form like AA
    #     RA metric punishes the high-degree common neighbos more heavily tahn A
    # RA performs better for the network with high average degrees
    def RA(self, i, j,mode="out"):
        if mode == "out":
            common_neighbors = self.common_neighbors(i, j)
            return sum([1 / self.outdegrees[item] for item in common_neighbors if self.outdegrees[item]])
        elif mode == "in":
            common_neighbors = self.common_neighbors(i,j,mode="in")
            return sum([1 / self.indegrees[item] for item in common_neighbors if self.indegrees[item]])
    '''
    path based metric
    '''
    # local path: make use of informtaion of local paths with length 2 and length 3
    def LP(self,alpha):
        return np.linalg.matrix_power(self.adjmatrix,2) + alpha*(np.linalg.matrix_power(self.adjmatrix,3))

    # Katz: katz metirc is based on ensemble of all paths
    # the paths are exponential dampled by length that can give more weiggts to the shorter paths
    # one can verify that the matrix of scores is given by (inverse of(I- beta*M))-I
    def katz(self,beta):
        return np.linalg.inv(np.identity(self.n) - beta*self.adjmatrix) -np.identity(self.n)

    # FindLink: reference paper: Fast and accurate link prediction in social networking systems
    # L is teh maximum length of paths explored in G
    # todo: fill the score
    def FL(self,L):
        # adjMatrix = self.adjmatrix
        adjMatrix = np.zeros((self.n,self.n),dtype=object)
        for i in range(self.n):
            for j in range(self.n):
                if self.adjmatrix[i,j] == 1:
                    adjMatrix[i,j] = str(j)
                else:
                    adjMatrix[i,j] = 0
        for count in range(2,L+1):
            for i in range(self.n):
                for j in range(self.n):
                    for k in range(self.n):
                        if adjMatrix[i,k] and adjMatrix[k,j]:
                            adjMatrix[i,j] = str(adjMatrix[i,k])+","+str(adjMatrix[k,j])
            # for i in range(self.n):
            #     for j in range(self.n):
            #         denominator = 1
            #         for k in range(1,i):
            #             denominator = denominator*(self.n - k)
            #     simScore[i,j] = simScore[i,j]+(1/(i-1))*
        return adjMatrix



prede = topology(g)
print(prede.eigen_cen())