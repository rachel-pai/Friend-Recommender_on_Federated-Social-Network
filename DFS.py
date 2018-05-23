from userInfo import *
from helper import *
## DFS (depth-first search)
#DFS is similair to BFS, expcept that, at each iteration,
# it visits an unvisited neighbor of most recentely visited nodes

def findNext(id,temp_ids,followingData,ids):
   print("id",id)
   if id == 0:
      vertice = getRandomlyUser(user_count)
   else:
      vertice = Node(id)
   # pay attention,account['following account'] isnt euqual to nodes amunt that following()return
   followingList = vertice.getFollowing()
   if vertice.id not in temp_ids:
      ids.append({'id': vertice.id, 'user': vertice.name, 'url': vertice.url})
   temp_ids.append(vertice.id)
   n = -1
   if len(followingList) == 0:  # donesnt have any following
      if id == 0:       # if is the sart node, then randomly choose start node again, this code does not converge
         return  0,temp_ids, followingData,ids
      else:
         ## if the node does not have any following children,
         ## look back into the parent children, until find node has folloing nodes
         while True:
            nextId = temp_ids[n]
            vertice = Node(nextId)
            n = n-1
            if abs(n) > len(temp_ids):
               return 0,temp_ids,followingData,ids
               # need randomly choose node again, since all parents dont have any following nodes
            if len(vertice.getFollowing()) > 1:
               break
   # we shouldnt look for visited children
   followingLIst = vertice.getFollowing()
   followinglistId = [x.id for x in followingLIst]
   unvisitedNodes = list(set(followinglistId) - set(temp_ids))
   followingItem = Node(random.choice(unvisitedNodes))
   followingItem.getParent(vertice.id)

   if len(followingItem.getFollowing()) == 0:
      # if the following node doenst have any children, we put its parente as the next iterate starting node
      nextId = followingItem.parentID
      ids.append({'id': vertice.id, 'user': vertice.name, 'url': vertice.url})
      temp_ids.append(vertice.id)
   else:
      nextId = followingItem.id
      ids.append({'id': followingItem.id, 'user': followingItem.name, 'url': followingItem.url})
      temp_ids.append(followingItem.id)

   followingData.append({"from": vertice.id, "to": followingItem.id})

   return nextId,temp_ids,followingData,ids

# depth first
def getFollowing(iterateNum):
   followingData = []
   temp_ids = []
   if os.path.exists('path.csv'):
      with open('path.csv','rt',encoding='utf-8') as csvfile:
         readCSV = csv.reader(csvfile, delimiter=',')
         temp_ids = [r[0] for r in readCSV]
         temp_ids.pop(0)
   ids = []
   for i in range(0,iterateNum):
      if i == 0:
         nextId, temp_ids, followingData, ids = findNext(0,temp_ids,followingData,ids)
      else:
         nextId, temp_ids, followingData, ids = findNext(nextId,temp_ids,followingData,ids)
   return followingData,ids

followingData,nodes = getFollowing(10)

writeIntoCsvFile(filename='path', header = ['id','user','url'],writenData=nodes)
writeIntoCsvFile(filename='edges', header = ['from','to'],writenData=followingData)
removeDuplicate('path','nodes')
