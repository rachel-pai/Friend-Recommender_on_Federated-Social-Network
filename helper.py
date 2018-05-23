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

   def getFollowing(self):
      NodeList = []
      followingList = mastodon.account_following(self.id)
      for item in  followingList:
         NodeList.append(Node(item['id']))
      return NodeList
   def getParent(self,id):
      self.parentID = id

def getRandomlyUser(num):
   while True:
      randomUser = random.randint(1, num + 1)
      vertice = Node(randomUser)
      if vertice.name:
         break
   return vertice



