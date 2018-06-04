# coding:utf-8
# Created by chen on 03/06/2018
# email: q.chen@student.utwente.nl

import igraph as ig
import louvain
G = ig.Graph.Famous('Zachary')
partition = louvain.find_partition(G,louvain.ModularityVertexPartition)
print(partition)
subsubgraphs = partition.subgraphs()
for c,subsubgraph in enumerate(subsubgraphs):
        ig.plot(subsubgraph)

