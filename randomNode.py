from userInfo import *
from helper import *
# randomly choose nodes without any rule

def getFollowing(num):
   followingData,temp_ids,ids = [],[],[]
   for i in range(0,num):
      vertice = getRandomlyUser(user_count)
      temp_ids.append(vertice.id)
      ids.append({'id': vertice.id, 'user': vertice.name, 'url': vertice.url})
      if vertice.followingCount:
            for x in vertice.getFollowing():
               followingData.append({"from": vertice.id,"to":x.id})
               if x.id not in temp_ids:
                  temp_ids.append(x.id)
                  ids.append({'id':x.id,'user':x.name,'url':x.url})
   return followingData,ids

followingData,nodes = getFollowing(50)

writeIntoCsvFile(filename='RR_node_temp', header = ['id','user','url'],writenData=nodes)
writeIntoCsvFile(filename='RR_edge', header = ['from','to'],writenData=followingData)
removeDuplicate('RR_node_temp','RR_node')
