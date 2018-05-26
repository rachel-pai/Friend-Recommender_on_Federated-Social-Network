# coding:utf-8
# Created by chen on 25/05/2018
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


class PrecalculatedStuff(object):
    def __init__(self, graph):
        self.graph = graph
        self.outdegrees = graph.outdegree()
        self.indegrees = graph.indegree()
        self.adjlistOut = list(map(set,graph.get_adjlist()))
        self.adjlistIn = list(map(set,graph.get_adjlist(mode=IN)))
        self.adjmatrix = np.matrix([graph.get_adjacency()[i] for i in range(len(graph.vs))])
        self.m = graph.ecount() #number of edges in the graph
        self.n = graph.vcount() # vertice number
    def laplacian_matrix(self):
        # return csgraph.laplacian(self.adjmatrix, normed=False,return_diag=True,use_out_degree=True)
        return csgraph.laplacian(self.adjmatrix, normed=False,use_out_degree=True)

    def pse_laplacian(self):
        return np.linalg.pinv(self.laplacian_matrix())
    # hitting time only for undirectd and unweightd graph
    # commute time: since the hitting time metric is not symmetric
    # commute time count the expected steps both from x to y and from y to x
    def CT(self,i,j):
        pseLaplacian = self.pse_laplacian()
        return self.m*(pseLaplacian[i, i] + pseLaplacian[j, j] - 2*pseLaplacian[i, j])
    # cosine similarity time (CST): calcualte the similarity of two vectors
    def CST(self,i,j):
        pseLaplacian = self.pse_laplacian()
        return pseLaplacian[i, j] / np.sqrt(pseLaplacian[i, i] * pseLaplacian[j, j])

    # SimRank: SimRank is defined in a self-constraint way
    # according to the asssumption that two nodes are similar if they are connected to similar nods
    # calculate simRank based on  outdegree
    # regular simRank is based on indegree
    def simRank(self, r=0.8, max_iter = 100, eps=1e-4,mode="out"):
        sim_prev = np.zeros(self.n)
        sim = np.identity(self.n)
        for i in range(max_iter):
            if np.allclose(sim,sim_prev,atol=eps): # iterate until converge
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
                            s_uv = sum([sim_prev[neighbor_u][neighbor_v] for neighbor_u,neighbor_v in itertools.product(self.adjlistOut[u],self.adjlistOut[v])])
                            sim[u][v] = (r * s_uv) / (self.outdegrees[u]*self.outdegrees[v])
            elif mode == "in":
                for u in range(self.n):
                    for v in range(self.n):
                        if u == v:  # if u and v is equal return 1
                            continue
                        if self.indegrees[u] == 0 or self.indegrees[v] == 0:
                            sim[u][v] = 0
                        else:
                            s_uv = sum([sim_prev[neighbor_u][neighbor_v] for neighbor_u, neighbor_v in itertools.product(self.adjlistIn[u],self.adjlistIn[v])])
                            sim[u][v] = (r * s_uv) / (self.indegrees[u]*self.indegrees[v])
        return sim

    # rooted PageRank (RPR): Rooted PageRank is a modification of PageRank,
    # the rank of a node in graph is proportiaonl to the probabilityu taht the node wil be reached through a random walk on the graph
    def RPR(self,factor=0.5):
        D_inv = np.diag(1./np.array([max(x,1) for x in self.outdegrees]))
        return (1-factor)*(np.linalg.inv(np.identity(self.n) - D_inv.dot(self.adjmatrix)))



# print(g.vcount())
# print(g.ecount())
# prede = PrecalculatedStuff(g)
# sim = prede.simRank()
# print(sim)
# print(prede.outdegrees)
# D = np.diag(1./np.array([max(x,1) for x in prede.outdegrees]))
