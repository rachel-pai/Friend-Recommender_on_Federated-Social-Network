#coding:utf-8
#Created by chen on 26/05/2018
#email: q.chen@student.utwente.nl

import pandas as pd

def mergeCSVs(fileName1, fileName2, mergedFileName):
    file1 = pd.read_csv( fileName1+".csv")
    file2 = pd.read_csv(fileName2+".csv")
    frames = [file1,file2]
    merged = pd.concat(frames)
    # remove duplicates
    merged.drop_duplicates(subset=None, inplace=True)
    # print(merged.head(10))
    merged.to_csv(mergedFileName+ ".csv", index=False)

# DFS, BFS, RW, RR, DS
# mergeCSVs("node1","anode4","node")
# mergeCSVs("DFS_node1","DS_node1","node1")
# mergeCSVs("DFS_edge1","DS_edge1","edge1")

# mergeCSVs("DFS_node_temp","DFS_node2","DFS_node")


