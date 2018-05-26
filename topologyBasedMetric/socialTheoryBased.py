# coding:utf-8
# Created by chen on 26/05/2018
# email: q.chen@student.utwente.nl


# import igraph
from igraph import *
import pandas
import numpy as np
from scipy.sparse import csgraph
import itertools

nodesListData = pandas.read_csv("DFS_node.csv")
reassignId =nodesListData["id"].tolist()
getDict = {elem:count for count,elem in enumerate(reassignId)}
# print(getDict)
# nodesListData = nodesListData.sort_values(by=['id'])
edgeListData = pandas.read_csv("DFS_edge.csv")

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



# print(g.vcount())
# print(g.ecount())
prede = PrecalculatedStuff(g)
# print(prede.betweenness_centrality())
# print(prede.common_neighbors(0,1))
# A = prede.weak_tie_based(cent="degree")
print(prede.eigen_cen())
# sim = prede.simRank()
# print(sim)
# print(prede.outdegrees)
# D = np.diag(1./np.array([max(x,1) for x in prede.outdegrees]))

from scipy.sparse import csr_matrix

# print(A.nonzero())
# print(A[0,40])