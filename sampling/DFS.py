# coding:utf-8
# Created by chen on 27/05/2018
# email: q.chen@student.utwente.nl

# coding:utf-8
# Created by chen on 22/05/2018
# email: q.chen@student.utwente.nl
from userInfo import *
from helper import *


## DFS (depth-first search)
# DFS is similair to BFS, expcept that, at each iteration,
# it visits an unvisited neighbor of most recentely visited nodes

def findNext(id, temp_ids, followingData, ids):
    print("coming into findNext!")
    if id == 0:
        vertice = getRandomlyUser(user_count)
    else:
        vertice = Node(id)

    followingList = vertice.getFollowing()
    ids.append({'id': vertice.id, 'user': vertice.name, 'url': vertice.url})
    temp_ids.append(vertice.id)

    if len(followingList) == 0:
        if id == 0:
            return 0, temp_ids, followingData, ids
        else:
            return -1, temp_ids, followingData, ids
    # we shouldnt look for visited children
    followinglistId = vertice.getFollowingId()
    unvisitedNodes = list(set(followinglistId) - set(temp_ids))
    if unvisitedNodes:
        followingLIst = vertice.getFollowing()
        followinglistId = [x.id for x in followingLIst]
        unvisitedNodes = list(set(followinglistId).difference(temp_ids))

        followingItem = Node(random.choice(unvisitedNodes))  # randomly choosen unvisited nodes
        followingItem.getParent(vertice.id)

        if len(followingItem.getFollowing()) == 0:
            return 0,temp_ids,followingData,ids
        else:
            nextId = followingItem.id
            ids.append({'id': followingItem.id, 'user': followingItem.name, 'url': followingItem.url})
            temp_ids.append(followingItem.id)
            followingData.append({"from": vertice.id, "to": followingItem.id})
            print("success add paths!")
        return nextId, temp_ids, followingData, ids
    else:
        return 0, temp_ids, followingData, ids

# depth first
def getFollowing(iterateNum):
    followingData,temp_ids,ids = [],[],[]
    for i in range(0, iterateNum):
        if i == 0:
            nextId = 0
        nextId, temp_ids, followingData, ids = findNext(nextId, temp_ids, followingData, ids)
        if nextId == -1:
            break
    return followingData, ids

# for i in range(100):
# followingData, nodes = getFollowing(20)
# print("beginning !!")
# writeIntoCsvFile(filename='../data/DFS_node_temp', header=['id', 'user', 'url'], writenData=nodes)
# writeIntoCsvFile(filename='../data/DFS_edge', header=['from', 'to'], writenData=followingData)
# removeDuplicate('../data/DFS_node_temp', '../data/DFS_node')
addMissingNode('../data/DFS_edge', '../data/DFS_node', header=['id', 'user', 'url'])
