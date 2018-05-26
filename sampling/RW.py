# coding:utf-8
# Created by chen on 22/05/2018
# email: q.chen@student.utwente.nl

from userInfo import *
from helper import *
# random walk (RW)
# a ramdom walk simply selects the next hop unifromly at random
# from among the jneighbos of the current node
def findNext(id,followingData,ids,stopped):
   if id == 0:
      vertice = getRandomlyUser(user_count)
   else:
      vertice = Node(id)
   ids.append({'id': vertice.id, 'user': vertice.name, 'url': vertice.url})
   followingLIst = vertice.getFollowing()

   if len(followingLIst) == 0:
      if id == 0:
         return 0, followingData, ids,0         #reselect a new node
      else:
         return id,followingData,ids,1          #converge

   ## uniformly randomly choose the next node
   random_index = randrange(0, len(followingLIst))
   followingItem = followingLIst[random_index]
   followingData.append({"from": vertice.id, "to": followingItem.id})
   # random walk doesnt care whether next node is visited or not
   nextId = followingItem.id
   nextNode = Node(nextId)
   ids.append({'id': nextNode.id, 'user': nextNode.name, 'url': nextNode.url})
   return nextId,followingData,ids,0

# random walk
def getFollowing(iterateNum):
   followingData = []
   ids = []
   for i in range(0,iterateNum):
      if i == 0:
         nextId, followingData, ids,stopped = findNext(0,followingData,ids,stopped=0)
      else:
         nextId, followingData, ids,stopped = findNext(nextId,followingData,ids,stopped=0)
      if stopped == 1:
         break
   return followingData,ids

for i in range(1):
   followingData,nodes = getFollowing(300)
   writeIntoCsvFile(filename='../data/RW_node_temp', header = ['id','user','url'],writenData=nodes)
   writeIntoCsvFile(filename='../data/RW_edge', header = ['from','to'],writenData=followingData)
   removeDuplicate('../data/RW_node_temp','../data/RW_node')