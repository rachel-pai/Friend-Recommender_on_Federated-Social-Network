#coding:utf-8
#Created by chen on 25/05/2018
#email: q.chen@student.utwente.nl

# import igraph
from igraph import *
import pandas
import numpy as np
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
        self.degrees = graph.outdegree()
        self.adjlist = list(map(set,graph.get_adjlist(mode="OUT")))
        self.n = graph.vcount() # vertices number
        self.adjmatrix = np.matrix([graph.get_adjacency()[i] for i in range(len(graph.vs))])

    # local path: make use of informtaion of local paths with length 2 and length 3
    def LP(self,alpha):
        return np.linalg.matrix_power(self.adjmatrix,2) + alpha*(np.linalg.matrix_power(self.adjmatrix,3))

    # Katz: katz metirc is based on ensemble of all paths
    # the paths are exponential dampled by length that can give more weiggts to the shorter paths
    # one can verify that the matrix of scores is given by (inverse of(I- beta*M))-I
    def katz(self,beta):
        return np.linalg.inv(np.identity(self.n) - beta*self.adjmatrix) -np.identity(self.n)

    # FindLink: reference paper:
    # L is teh maximum length of paths explored in G
    def FL(self,L):
        # adjMatrix = self.adjmatrix
        adjMatrix = np.zeros((self.n,self.n),dtype=object)
        adjMatrix.astype(object)
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
prede = PrecalculatedStuff(g)
print(prede.adjmatrix)
# print(prede.adjmatrix[1,0])
# print(prede.katz(0.3))
# A = prede.adjmatrix
# A = np.matrix([[0,1,0,1,1],[1,0,1,1,0],[0,1,0,0,0],[1,1,0,0,1],[1,0,0,1,0]])
# print(np.linalg.matrix_power(A,3))
print(prede.FL(2))