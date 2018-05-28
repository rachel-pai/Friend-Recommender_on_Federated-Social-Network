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

# mergeCSVs("DFS_edge","DFS_edge2","DFS_edge")
# mergeCSVs("DFS_node_temp","DFS_node2","DFS_node")
#
# mergeCSVs("RW_edge","RW_edge2","RW_edge")
# mergeCSVs("RW_node","RW_node2","RW_node")
# mergeCSVs("DFS_edge","BFS_edge","all_edge")
# mergeCSVs("DFS_node","BFS_node","all_node")

# mergeCSVs("DFS_node_temp1","DFS_node_temp2","DFS_node5")

