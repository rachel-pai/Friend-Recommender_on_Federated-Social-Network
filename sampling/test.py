# # # # # coding:utf-8
# # # # # Created by chen on 25/05/2018
# # # # # email: q.chen@student.utwente.nl
# # # # from igraph import *
# # # # import numpy as np
# # # #
# # # # # Create the graph
# # # # vertices = [i for i in range(7)]
# # # # edges = [(0, 2), (0, 1), (0, 3), (1, 0), (1, 2), (1, 3), (2, 0), (2, 1), (2, 3), (3, 0), (3, 1), (3, 2), (2, 4), (4, 5),
# # # #          (4, 6), (5, 4), (5, 6), (6, 4), (6, 5)]
# # # #
# # # # g = Graph(vertex_attrs={"label": vertices}, edges=edges, directed=True)
# # # #
# # # # visual_style = {}
# # # #
# # # # # Scale vertices based on degree
# # # # outdegree = g.outdegree()
# # # # visual_style["vertex_size"] = [x / max(outdegree) * 25 + 50 for x in outdegree]
# # # #
# # # # # Set bbox and margin
# # # # visual_style["bbox"] = (800, 800)
# # # # visual_style["margin"] = 100
# # # #
# # # # # Define colors used for outdegree visualization
# # # # colours = ['#fecc5c', '#a31a1c']
# # # #
# # # # # Order vertices in bins based on outdegree
# # # # bins = np.linspace(0, max(outdegree), len(colours))
# # # # digitized_degrees = np.digitize(outdegree, bins)
# # # #
# # # # # Set colors according to bins
# # # # g.vs["color"] = [colours[x - 1] for x in digitized_degrees]
# # # #
# # # # # Also color the edges
# # # # for ind, color in enumerate(g.vs["color"]):
# # # #     edges = g.es.select(_source=ind)
# # # #     edges["color"] = [color]
# # # #
# # # # # Don't curve the edges
# # # # visual_style["edge_curved"] = False
# # # #
# # # # # Community detection
# # # # communities = g.community_edge_betweenness(directed=True)
# # # # clusters = communities.as_clustering()
# # # #
# # # # # Set edge weights based on communities
# # # # weights = {v: len(c) for c in clusters for v in c}
# # # # g.es["weight"] = [weights[e.tuple[0]] + weights[e.tuple[1]] for e in g.es]
# # # #
# # # # # Choose the layout
# # # # N = len(vertices)
# # # # visual_style["layout"] = g.layout_fruchterman_reingold(weights=g.es["weight"], maxiter=1000, area=N ** 3,
# # # #                                                        repulserad=N ** 3)
# # # #
# # # # # Plot the graph
# # # # plot(g, **visual_style)
# # #
# # import numpy as np
# # # A = [1,2,3]
# # # B = [7,8]
# # # print(np.concatenate([A,B]))
# # #
# #
# # matrixOri = np.matrix([[0,1,1,0,1,1,0,1,0],
# #                       [1,0,0,1,0,0,0,0,0],
# #                       [1,0,0,1,0,0,0,0,0],
# #                       [0,1,1,0,0,0,0,0,1],
# #                       [1,0,0,0,0,0,1,0,0],
# #                       [1,0,0,0,0,0,1,0,0],
# #                       [0,0,0,0,1,1,0,0,0],
# #                       [1,0,0,0,0,0,0,0,1],
# #                       [0,0,0,1,0,0,0,1,0]])
# # print(matrixOri.shape)
# # nodeLen = 9
# # def FL(L):
# #     # adjMatrix = matrixOri
# #     adjMatrix = np.zeros((nodeLen, nodeLen), dtype=object)
# #     # adjMatrix.astype(object)
# #     for i in range(nodeLen):
# #         for j in range(nodeLen):
# #             if matrixOri[i, j] == 1:
# #                 adjMatrix[i, j] = str(i)
# #             else:
# #                 adjMatrix[i, j] = 0
# #     print(adjMatrix)
# #     for count in range(2, L + 1):
# #         for i in range(nodeLen):
# #             for j in range(nodeLen):
# #                 for k in range(nodeLen):
# #                     if adjMatrix[i, k] and adjMatrix[k, j]:
# #                         adjMatrix[i, j] = str(adjMatrix[i, k]) + "," + str(adjMatrix[k, j])
# #         # for i in range(nodeLen):
# #             # for items in adjMatrix[i]:
# #
# #
# #         # for i in range(nodeLen):
# #         #     for j in range(nodeLen):
# #         #         denominator = 1
# #         #         for k in range(1,i):
# #         #             denominator = denominator*(nodeLen - k)
# #         #     simScore[i,j] = simScore[i,j]+(1/(i-1))*
# #     return adjMatrix
# #
# # # 1,3,4,2=>0,2,3,1
# # print(FL(2)[0][3])
# #
# #
# #
# import itertools
# import numpy as np
# from igraph import *
# import pprint
# g = Graph([(1,2), (1, 4), (2,3), (3,1), (4,5), (5,4)],directed=True)
# nodelen = g.vcount()
# print("node count",nodelen)
# degrees = g.indegree()
# adjlist = list(map(set,g.get_adjlist(mode=IN)))
# adjlistMatrix = np.matrix([g.get_adjacency()[i] for i in range(len(g.vs))])
# def simRank(graph, r=0.8, max_iter=100, eps=1e-4):
#     sim_prev = np.zeros(nodelen)
#     sim = np.identity(nodelen)
#     for i in range(max_iter):
#         if np.allclose(sim, sim_prev, atol=eps):  # iterate until converge
#             break
#         sim_prev = np.copy(sim)
#         # for u,v in itertools.product(nodelen,nodelen):
#         for u in range(1, nodelen):
#             for v in range(1, nodelen):
#                 if u == v:  # if u and v is equal return 1
#                     continue
#                 s_uv = 0.0
#                 if degrees[u] == 0 or degrees[v] == 0:
#                     sim[u][v] = 0
#                 else:
#                     # for neighbor_u in adjlist[u]:
#                     #     for neighbor_v in adjlist[v]:
#                     #         s_uv += sim_prev[neighbor_u][neighbor_v]
#                     s_uv = sum([sim_prev[neighbor_u][neighbor_v] for neighbor_u, neighbor_v in
#                                 itertools.product(adjlist[u], adjlist[v])])
#                     sim[u][v] = (r * s_uv) / (degrees[u] * degrees[v])
#     return sim
#
# print(simRank(g).round(3))
# # calculate the in degree
# # print("adjlist matrix",adjlistMatrix[:,1])

import numpy as np
# print(np.ones([3,3]))
# import random
# print(random.randint(1,5))
# import numpy as np
# a = np.array([1,2,3,4])
# b = np.array([2,5])
# print(list(set(a).difference(b)))
