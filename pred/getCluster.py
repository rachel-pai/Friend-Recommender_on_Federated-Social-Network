# coding:utf-8
# Created by chen on 11/06/2018
# email: q.chen@student.utwente.nl

import louvain
import os
# import igraph
from igraph import *
from topology import *
import numpy as np
import pickle
import pandas

def getCluster(test=0,edgeFile='aedge.csv',nodeFile = 'anode.csv'):
    nodesListData = pandas.read_csv(nodeFile)
    reassignId = nodesListData["id"].tolist()
    getDict = {elem: count for count, elem in enumerate(reassignId)}
    edgeListData = pandas.read_csv(edgeFile)

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

    # save the rearranged igraph
    g.save('../pred/newTestGraph.net', format="gml")
    with open('../pred/newTestGraph', 'wb') as fp:
            pickle.dump(g, fp)


    ''''
    need c components: community_optimal_modularity
    todo: ask Dai to run this method
    '''
    # infomap_clutsers = g.community_infomap()
    infomap_clutsers = louvain.find_partition(g, louvain.ModularityVertexPartition)
    # get the custering graph
    clusters = infomap_clutsers.subgraphs()
    # save clusters into csv files
    # run once time
    for n,graphs in enumerate(clusters):
        if graphs.vcount() > 50:
            if graphs.vcount() > 300:
                # different with community_edge_betweenness()
                # subsubgraph = graphs.community_walktrap().as_clustering()
                # subsubgraph = graphs.community_infomap()
                # subsubgraph = louvain.find_partition(graphs, louvain.ModularityVertexPartition)
                subsubgraph = graphs.community_infomap()
                subsubgraphs = subsubgraph.subgraphs()
                for c,subsubgraph in enumerate(subsubgraphs):
                    if subsubgraph.vcount() > 50:
                        with open('../pred/cluster/test/cluster_sub_'+str(c), 'wb') as fp:
                            pickle.dump(subsubgraph, fp)
            else:
                with open('../pred/cluster/test/cluster_' + str(n), 'wb') as fp:
                    pickle.dump(graphs, fp)