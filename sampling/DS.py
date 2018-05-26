# coding:utf-8
# Created by chen on 24/05/2018
# email: q.chen@student.utwente.nl
from userInfo import *
from helper import *

## SEC (sample edge counát)
# given the currently constructed sample S
# how can we select a node v with the highest degree without having knowledfe of N?
# the highest degree without having knowldege of N?
# The SEC strategy tracks the links from the currently constructed
# sample S to each node v ∈ N(S) and selects the
# node v with the most links from S. In other words, we
# use the degree of v in the induced subgraph of S ∪ {v} as
# an approximation of the degree of v in the original network G

## randomly choose a node
# find the node witth  the highest followings

def findNext(nodeId, temp_dict,followingData,ids):
    if nodeId == 0:
        vertices = getRandomlyUser(nodeId)
    else:
        vertices = Node(nodeId)
    nodeDegree = vertices.followingCount
    ids.append({'id': vertices.id, 'user': vertices.name, 'url': vertices.url})
    temp_dict.append({'id':vertices.id,'following':nodeDegree})
    if nodeDegree == 0:
        # findNext(0,temp_dict,followingData,ids)
        print("nodeDegree is zero")
        return 0,temp_dict,followingData,ids
    else:
        followingList = vertices.getFollowing()
        for followingNode in followingList:
            followingData.append({"from": vertices.id, "to": followingNode.id})
            ids.append({'id': followingNode.id, 'user': followingNode.name, 'url': followingNode.url})
            temp_dict.append({'id': followingNode.id, 'following': followingNode.followingCount})
        temp_dict =[i for n, i in enumerate(temp_dict) if i not in temp_dict[n + 1:]]   # remove dupilcates
        nextNodeId = sorted(temp_dict, key=itemgetter('following'))[-1]['id']
        # findNext(nextNodeId,temp_dict,followingData,ids)
    return nextNodeId,temp_dict,followingData,ids

# SEC
def getFollowing(iterateNum):
   followingData = []
   temp_ids = []
   ids = []
   for i in range(0,iterateNum):
      if i == 0:
         nextId, temp_ids, followingData, ids = findNext(0,temp_ids,followingData,ids)
      else:
         nextId, temp_ids, followingData, ids = findNext(nextId,temp_ids,followingData,ids)
   return followingData,ids

followingData,nodes = getFollowing(10)

writeIntoCsvFile(filename='../data/DS_node_temp', header = ['id','user','url'],writenData=nodes)
writeIntoCsvFile(filename='../data/DS_edge', header = ['from','to'],writenData=followingData)
removeDuplicate('../data/DS_node_temp','../data/DS_node')
