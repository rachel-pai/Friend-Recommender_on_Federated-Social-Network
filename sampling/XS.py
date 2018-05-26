# coding:utf-8
# Created by chen on 24/05/2018
# email: q.chen@student.utwente.nl

from userInfo import *
from helper import *

#XS: (expansion sampling)
# The XS strategy is based on the conceot of expansion from work on expander graphs
# and seeks to greedily construct the sample with teh maximal expansion:
# argmax s:|s| = k*(|N(S)|/|S|)
# where k is teh desired samole size[23,40]
# at each iteration, the next node v selected from inclusion in the sample is chosen
# based on the expression argmax|N({v}) - (N(S)U S)|

def findNext(nodeId, temp_dict,followingData,ids):
    if nodeId == 0:
        vertices = getRandomlyUser(nodeId)
    else:
        vertices = Node(nodeId)
    nodeDegree = vertices.followingCount
    ids.append({'id': vertices.id, 'user': vertices.name, 'url': vertices.url})
    temp_dict.append(vertices.id)
    if nodeDegree == 0:
        return 0,temp_dict,followingData,ids
    else:
        followingList = vertices.getFollowing()
        scoreMax = 0
        for followingNode in followingList:
            score = abs(followingNode.followingCount - max(vertices.followingCount,len(list(set(temp_dict)))))
            print("score and maxScore",score,scoreMax)
            if score > scoreMax:
                selectedNode = followingNode.copy()
                scoreMax = score
        followingData.append({"from": vertices.id, "to": selectedNode.id})
        ids.append({'id': selectedNode.id, 'user': selectedNode.name, 'url': selectedNode.url})
        temp_dict.append(selectedNode.id)
        nextNodeId = selectedNode.id
    return nextNodeId,temp_dict,followingData,ids

# xs
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

for i in range(10):
    followingData,nodes = getFollowing(3)
writeIntoCsvFile(filename='../data/XS_node_temp', header = ['id','user','url'],writenData=nodes)
writeIntoCsvFile(filename='../data/XS_edge', header = ['from','to'],writenData=followingData)
removeDuplicate('../data/XS_node_temp','../data/XS_node')
addMissingNode('../data/XS_edge','../data/XS_node',header=['id','user','url'])
