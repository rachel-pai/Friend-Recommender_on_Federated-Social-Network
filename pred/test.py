# coding:utf-8
# Created by chen on 03/06/2018
# email: q.chen@student.utwente.nl

import numpy as np
import pickle
from sklearn.decomposition import PCA
import os
# import igraph
from igraph import *
import pandas
from topology import *
import numpy as np
import pickle
from scipy.sparse import csgraph
import itertools
import pickle
import louvain

nodesListData = pandas.read_csv("../data/DS_node.csv")
reassignId = nodesListData["id"].tolist()
getDict = {elem: count for count, elem in enumerate(reassignId)}
edgeListData = pandas.read_csv("../data/DS_edge.csv")

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
print(g.vs[1]["label"])

# infomap_clutsers = g.community_infomap()
# layout = g.layout("kk")
# plot(infomap_clutsers.membership, "graph.pdf", layout=layout)
# print(max(infomap_clutsers.membership))

# louvains = louvain.find_partition(g, louvain.ModularityVertexPartition)
# layout = g.layout("kk")
# plot(louvains.membership, "louvains.pdf", layout=layout)

import igraph
# g = igraph.Graph.Barabasi(n = 20, m = 1)
# i = g.community_infomap()
i = louvain.find_partition(g, louvain.ModularityVertexPartition)
cleanM = [n for n in i.membership if i.membership.count(n)>50]
print("cleanM",len(cleanM))
removedlist  = set(i.membership)-set(cleanM)
removedIndex =[i for i,x in enumerate(i.membership) if x in list(removedlist)]
# removedIndex = i.membership.index(list(removedlist))
g.delete_vertices(removedIndex)
g.delete_vertices(g.vs.select(_degree=0))
pal = igraph.drawing.colors.ClusterColoringPalette(len(set(cleanM)))
g.vs['color'] = pal.get_many(cleanM)
# igraph.plot(g)
# layout_kamada_kawai
igraph.plot(g,'all_louvain.png',vertex_label=None,bbox=(1024,1024))
