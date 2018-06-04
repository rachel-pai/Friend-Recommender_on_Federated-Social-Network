# coding:utf-8
# Created by chen on 28/05/2018
# email: q.chen@student.utwente.nl

# coding:utf-8
# Created by chen on 28/05/2018
# email: q.chen@student.utwente.nl

# find clusteriing of data
import louvain

def collectFeature(graph,cID):
    '''
    cid: cid is cluter id
    '''
    prede = topology(graph)
    flatten_matrix = prede.adjmatrix.flatten()  # label
    with open('../pred/temp/y_labels_'+str(cID), 'wb') as fp:
        pickle.dump(flatten_matrix, fp)

    col_and_row = np.unravel_index(list(range(prede.n * prede.n)), (prede.n, prede.n))
    ## get the corresponding index of the flatten matrix
    all_matrix = []
    for i, j in zip(col_and_row[0], col_and_row[1]):
        all_matrix.append([i, j])

    # for each i,j pair get the corresponding score
    score_matirx = []
    # score in whole nodes
    RPR_socre = prede.RPR()
    simRank_score = prede.simRank()
    katz_score = prede.katz(beta=0.9)
    aa_score = prede.AA_matrix()
    tweak_score = prede.weak_tie_based()
    for item in all_matrix:
        i = item[0]
        j = item[1]
        item_score = []
        item_score.append(RPR_socre[i, j])
        item_score.append(simRank_score[i, j])
        item_score.append(katz_score[i, j])
        item_score.append(aa_score[i, j])
        item_score.append(tweak_score[i, j])
        item_score.append(prede.CT(i, j))
        item_score.append(prede.CST(i, j))
        # item_score.append(prede.degree_product(i, j))
        item_score.append(prede.CN(i, j))
        # item_score.append(prede.JC(i, j))
        # item_score.append(prede.SI(i, j))
        item_score.append(prede.SC(i, j))
        item_score.append(prede.HP(i, j))
        # item_score.append(prede.HD(i, j))
        item_score.append(prede.LHN(i, j))
        item_score.append(prede.PD(i, j))
        # item_score.append(prede.AA(i,j))
        # item_score.append(prede.PA(i, j))
        item_score.append(prede.RA(i, j))
        score_matirx.append(item_score)

    with open('../pred/temp/score_matirx_'+str(cID), 'wb') as fp:
        pickle.dump(score_matirx, fp)

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

nodesListData = pandas.read_csv("all_node.csv")
reassignId = nodesListData["id"].tolist()
getDict = {elem: count for count, elem in enumerate(reassignId)}
edgeListData = pandas.read_csv("all_edge.csv")

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

''''
need c components: community_optimal_modularity 
todo: ask Dai to run this method 
'''
infomap_clutsers = g.community_infomap()
# get the custering graph
clusters = infomap_clutsers.subgraphs()
# save clusters into csv files
# run once time
for n,graphs in enumerate(clusters):
    if graphs.vcount() > 50:
        if graphs.vcount() > 500:
            # different with community_edge_betweenness()
            # subsubgraph = graphs.community_walktrap().as_clustering()
            # subsubgraph = graphs.community_infomap()
            subsubgraph = louvain.find_partition(graphs, louvain.ModularityVertexPartition)
            subsubgraphs = subsubgraph.subgraphs()
            for c,subsubgraph in enumerate(subsubgraphs):
                if subsubgraph.vcount() > 50:
                    subsubgraph.save('../pred/cluster/cluster_sub_'+str(c)+'.net')
        else:
            graphs.save('../pred/cluster/cluster_' + str(n) + '.net')


graphsList = []
directory = '../pred/cluster/'
for filename in os.listdir(directory):
    if filename.endswith(".net"):
        graphsList.append(load(os.path.join(directory, filename)))

score_matrix,flatten_matrix = [],[]
for c,graph in enumerate(graphsList):
    collectFeature(graph,c)

