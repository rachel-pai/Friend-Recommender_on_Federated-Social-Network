# coding:utf-8
# Created by chen on 25/05/2018
# email: q.chen@student.utwente.nl

# import igraph
from igraph import *
import pandas
import numpy as np
nodesListData = pandas.read_csv("BFS_node.csv")
reassignId =nodesListData["id"].tolist()
getDict = {elem:count for count,elem in enumerate(reassignId)}
# print(getDict)
# nodesListData = nodesListData.sort_values(by=['id'])
edgeListData = pandas.read_csv("BFS_edge.csv")

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
        self.adjlist = list(map(set,graph.get_adjlist()))

    def degree_product(self,i,j):
        return self.degrees[i] * self.degrees[j]

    def common_neighbors(self, i, j):
        return self.adjlist[i].intersection(self.adjlist[j])

    def all_neighbors(self,i,j):
        return self.adjlist[i].union(self.adjlist[j])

    # CN: common neighborhoods
    def CN(self,i,j):
        return len(self.common_neighbors(i,j))

    # jaccard_coeff: normalize the size of common neighbors
    def JC(self,i,j):
        return 1.0*len(self.common_neighbors(i,j)) / len(self.all_neighbors(i,j))

    #soresen_index: besides considering the size of the common neighbors, it also points out lower degrees of nodes wouldj have higher link likelihood
    def SI(self,i,j):
        return 1.0*len(self.common_neighbors(i,j)) /(self.degrees[i] + self.degrees[j])

    #salton_cosin_sim: cosine matric
    def SC(self,i,j):
        return 1.0*len(self.common_neighbors(i,j)) / np.sqrt(self.degrees[i] * self.degrees[j])
    #Hub_Promoted: HP defines the topological overlap of nodex and y
    # the HP value is deterined by the lower degree of nodes
    def HP(self,i,j):
        return 1.0*len(self.common_neighbors(i,j)) / np.minimum(self.degrees[i],self.degrees[j])

    #Hub_depressed: hub depressed, similar to HP, but the value is determined by the higher degree of nodes
    def HD(self,i,j):
        return 1.0*len(self.common_neighbors(i,j))/np.maximum(self.degrees[i],self.degrees[j])

    #Leicht_holem_Nerman: assign high similarity to node pairs that have many common neighbors compared not to the possible maximum
    # but to the expected number of such neighbors
    def LHN(self,i,j):
        return 1.0*len(self.common_neighbors(i,j))/(self.degrees[i]*self.degrees[j])

    #Parameter_dependent: lamda is a free parameter, when lambda = 0, PD degenerates to CN
    # if lambda = 0.1 and lambda =1, it degenerates to Salton and LHN metric, respectively
    def  PD(self,i,j,lam):
        return 1.0*len(self.common_neighbors(i,j))/np.power(self.degrees[i]*self.degrees[j],lam)

    # AA: widely used in social network
    # the AA measure is formulated related to Jaccard's coefficent
    # but here, common neighbos which have fewer neighbors are weighted more heavily
    def AA(self,i,j):
        sum = 0
        common_neighbors = self.common_neighbors(i,j)
        for item in common_neighbors:
            if self.degrees[item] == 0:
                continue
            else:
                sum = sum + 1/np.log(self.degrees[item])
        return sum

    # preferential_attachemnt: preferential attachement:
    # the PA metric indeicates that new lnks will be more likely to connect high-degree nodes than lower ones
    def PA(self,i,j):
        return self.degrees[i] * self.degrees[j]

    # Resource_Allocation: motivated by the physical processes of resource allocatiojn
    # RA metric has a similar form like AA
#     RA metric punishes the high-degree common neighbos more heavily tahn A
    # RA performs better for the network with high average degrees
    def RA(self,i,j):
        sum = 0
        common_neighbors = self.common_neighbors(i,j)
        for item in common_neighbors:
            try:
                sum = sum + 1/self.degrees[item]
            except ZeroDivisionError:
                continue
        return sum


prede = PrecalculatedStuff(g)
commons = prede.adjlist[1]
print(commons)
# scores = {}
# scores["CN"] = prede.CN(1,2)
# scores["JC"] = prede.JC(1,2)
# scores["SI"] = prede.SI(1,2)
# scores["SC"] = prede.SC(1,2)
# scores["HP"] = prede.HP(1,2)
# scores["HD"] = prede.HD(1,2)
# scores["LHN"] = prede.LHN(1,2)
# scores["PD"] = prede.PD(1,2,0.8)
# scores["AA"] = prede.AA(1,2)
# scores["PA"] = prede.PA(1,2)
# scores["RA"] = prede.RA(1,2)
# print(scores)
