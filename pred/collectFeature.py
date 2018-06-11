# coding:utf-8
# Created by chen on 28/05/2018
# email: q.chen@student.utwente.nl

# coding:utf-8
# Created by chen on 28/05/2018
# email: q.chen@student.utwente.nl

# find clusteriing of data
import louvain
import os
# import igraph
from igraph import *
from topology import *
import numpy as np
import pickle

with open('../pred/newTestGraph', 'rb') as score_file:
    score_file.seek(0)
    wholeGraph = pickle.load(score_file)

def collectFeature(graph,cID):
    '''
    cid: cid is cluter id
    '''
    prede = topology(graph)
    flatten_matrix = prede.adjmatrix.flatten()  # label
    flatten_matrix = []

    flatten_matrix_c = []
    col_and_row = np.unravel_index(list(range(prede.n * prede.n)), (prede.n, prede.n))
    ## get the corresponding index of the flatten matrix
    all_matrix = []
    for i, j in zip(col_and_row[0], col_and_row[1]):
        all_matrix.append([i, j])
    for i, j in zip(col_and_row[0], col_and_row[1]):
        labelFrom = graph.vs[i]["label"]
        labelTo = graph.vs[j]["label"]
        fromID = wholeGraph.vs(label_eq=labelFrom)
        toID = wholeGraph.vs(label_eq=labelTo)
        fromID = fromID[0].index
        toID = toID[0].index
        # flatten_matrix_c.append({"fromC":graph.vs[i]["label"],"toC":graph.vs[j]["label"],"fromW":wholeGraph.vs[fromID]["label"],
        #                          "toW":wholeGraph.vs[toID]["label"],"edge":wholeGraph.get_eid(fromID, toID, directed=False, error=False)})
        flatten_matrix.append(wholeGraph.get_eid(fromID, toID, directed=False, error=False))
        # search in the whole graph ,whether connected
    with open('../pred/temp/test/y_labels_' + str(cID), 'wb') as fp:
        pickle.dump(flatten_matrix, fp)
    # import json
    # with open('outputfile', 'w') as fout:
    #     json.dump(flatten_matrix_c, fout)
    # print(flatten_matrix)

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
        print("start RPR")
        item_score.append(simRank_score[i, j])
        print("start simRank")
        item_score.append(katz_score[i, j])
        print("start katz")
        item_score.append(aa_score[i, j])
        print("start aa")
        item_score.append(tweak_score[i, j])
        print("start tweak")
        item_score.append(prede.CT(i, j))
        print("start ct")
        item_score.append(prede.CST(i, j))
        print("start cst")
        # item_score.append(prede.degree_product(i, j))
        item_score.append(prede.CN(i, j))
        print("stat cn")
        # item_score.append(prede.JC(i, j))
        # item_score.append(prede.SI(i, j))
        item_score.append(prede.SC(i, j))
        print("start sc")
        item_score.append(prede.HP(i, j))
        print("start hp")
        # item_score.append(prede.HD(i, j))
        item_score.append(prede.LHN(i, j))
        print("start LHN")
        item_score.append(prede.PD(i, j))
        print("start pd")
        # item_score.append(prede.AA(i,j))
        # item_score.append(prede.PA(i, j))
        item_score.append(prede.RA(i, j))
        print("start RA")
        score_matirx.append(item_score)

    with open('../pred/temp/test/score_matirx_'+str(cID), 'wb') as fp:
        pickle.dump(score_matirx, fp)



graphsList = []
directory = '../pred/cluster/test/'
for filename in os.listdir(directory):
    with open(os.path.join(directory, filename), 'rb') as score_file:
        score_file.seek(0)
        temp= pickle.load(score_file)
    graphsList.append(temp)

score_matrix,flatten_matrix = [],[]
for c,graph in enumerate(graphsList):
    if c>12:
        collectFeature(graph,c)

