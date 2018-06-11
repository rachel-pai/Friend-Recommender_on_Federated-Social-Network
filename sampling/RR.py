# coding:utf-8
# Created by chen on 22/05/2018
# email: q.chen@student.utwente.nl

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

for i in range(3):
   followingData,nodes = getFollowing(50)
   writeIntoCsvFile(filename='../data/RR_node_temp_2', header = ['id','user','url'],writenData=nodes)
   writeIntoCsvFile(filename='../data/RR_edge_2', header = ['from','to'],writenData=followingData)
   # removeDuplicate('../data/RR_node_temp','../data/RR_node')
