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
      while True:
         vertice = getRandomlyUser(user_count)
         if vertice.followingCount:
            break
   else:
      vertice = Node(id)
   print("come into find next function! ")
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
               print("try item in queue")
               for id in item:
                  q,temp_ids,followingData,ids = findNext(id,q,temp_ids,followingData,ids)
            except Empty:
               break
   return followingData,ids

followingData,nodes = getFollowing(3)


writeIntoCsvFile(filename='BFS_node_temp', header = ['id','user','url'],writenData=nodes)
writeIntoCsvFile(filename='BFS_edge', header = ['from','to'],writenData=followingData)
removeDuplicate('BFS_node_temp','BFS_node')

addMissingNode('BFS_edge','BFS_node',header=['id','user','url'])
