# coding:utf-8
# Created by chen on 22/05/2018
# email: q.chen@student.utwente.nl
from userInfo import *
from helper import *
# BFS: breadth first search
# starting wit ha single seed node
# the BFS explores teh neighbors of visited nodes
# at each iteration,
# it raverses an unvisited neighbor of the earliest visited node
# BFS is biased toward high-degree and high-PageRank nodes
# use Queue

def findNext(id,q,temp_ids,followingData,ids):
   if id == 0:
       vertice = getRandomlyUser(user_count)
   else:
      vertice = Node(id)

   neighbors = []
   temp_ids.append(vertice.id)
   # if vertice.id not in temp_ids:
   ids.append({'id': vertice.id, 'user': vertice.name, 'url': vertice.url})
   followingList = vertice.getFollowing()
   if len(followingList) > 0:
      for followingNode in followingList:
         followingData.append({"from": vertice.id, "to": followingNode.id})
         neighbors.append(followingNode.id)
         ids.append({'id': followingNode.id, 'user': followingNode.name, 'url': followingNode.url})
         temp_ids.append(followingNode.id)
      temp_ids = list(set(temp_ids))
      neighbors = list(set(neighbors))  # remove dupilicates
      q.put(neighbors)
      print("neighbor",neighbors)

   return q,temp_ids,followingData,ids

# dia is graph diameters you want
def getFollowing(num):
   # get all followings, num means how many users you want get
   followingData,temp_ids,ids,randomUser = [],[],[],[]
   for count in range(0,num):
      if count == 0: #first user
         q = Queue()
         q, temp_ids, followingData, ids = findNext(0,q,temp_ids,followingData,ids)
      else:
         old_q = q
         q = Queue()
         while True:
            try:
               item = old_q.get(block=False)
               for id in item:
                  q,temp_ids,followingData,ids = findNext(id,q,temp_ids,followingData,ids)
            except Empty:
               break
   return followingData,ids

followingData,nodes = getFollowing(1)


writeIntoCsvFile(filename='../data/BFS_node_temp', header = ['id','user','url'],writenData=nodes)
writeIntoCsvFile(filename='../data/BFS_edge', header = ['from','to'],writenData=followingData)
removeDuplicate('../data/BFS_node_temp','../data/BFS_node')
addMissingNode('../data/BFS_edge','../data/BFS_node',header=['id','user','url'])
