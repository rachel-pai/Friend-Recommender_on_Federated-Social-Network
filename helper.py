from userInfo import *
from bson import json_util
import json
import csv
import random
import os.path
from random import randrange
from operator import itemgetter
from mastodon import Mastodon
from queue import *
from more_itertools import unique_everseen
import copy
class Node:
   def __init__(self,id):
      self.id = id
      try:
         self.account = mastodon.account(id)
         self.name = self.account['username']
         self.url = self.account['url']
         self.followingCount = self.account['following_count']
      except:
         self.account = None
         self.name = None
         self.url = None
         self.followingCount = None

   def copy(self):
      return Node(self.id)
   def getFollowing(self):
      NodeList = []
      followingList = mastodon.account_following(self.id)
      for item in  followingList:
         NodeList.append(Node(item['id']))
      return NodeList
   def getParent(self,id):
      self.parentID = id

def writeIntoCsvFile(filename,header,writenData):
   writeHead = True
   if os.path.exists(filename+'.csv'):
      writeHead = False
   with open(filename+'.csv', 'a',newline='') as fileCsv:
      header = header
      writer = csv.DictWriter(fileCsv, delimiter=',', fieldnames=header)
      if writeHead:
         writer.writeheader()
      for rawData in writenData:
         writer.writerow(rawData)

def removeDuplicate(fileTempName, fileName):
   with open(fileTempName+'.csv', 'r') as in_file, open(fileName+'.csv', 'w') as out_file:
      seen = set()  # set for fast O(1) amortized lookup
      for line in in_file:
         if line in seen: continue  # skip duplicate
         seen.add(line)
         out_file.write(line)


def addMissingNode(edgeFile, nodeFile,header):
   if os.path.exists(edgeFile+'.csv'):
      with open(edgeFile+'.csv', 'rt', encoding='utf-8') as edgefile:
         readCSV = csv.reader(edgefile, delimiter=',')
         edgeId = [r for r in readCSV]
         edgeId.pop(0)
      edgeId = [r[0] for r in edgeId] + [r[1] for r in edgeId]
      result = list(set(edgeId))

   if os.path.exists(nodeFile+'.csv'):
      with open(nodeFile+'.csv','rt',encoding='utf-8') as csvfile:
         readCSV = csv.reader(csvfile, delimiter=',')
         nodeId = [r[0] for r in readCSV]
         nodeId.pop(0)
      nodeId = list(set(nodeId))

   missingIds = list(set(result)-set(nodeId))
   ids =[]
   for missingId in missingIds:
      node_temp = Node(missingId)
      ids.append({'id': node_temp.id, 'user': node_temp.name, 'url': node_temp.url})
   if ids:
      writeIntoCsvFile(nodeFile,header, ids)


def getRandomlyUser(num):
   while True:
      randomUser = random.randint(1, num + 1)
      vertice = Node(randomUser)
      if vertice.name:
         break
   return vertice


