from userInfo import *
from helper import *


# userInfo = mastodon.account(62571)
# print(userInfo['username'])
# rand_item = random.choice(['a','m','o'])
# print(rand_item)

# a = [1,2,3,4,10]
# print(len(a))
# print(a[-6])

list_to_be_sorted = [{'name':'Bart', 'age':1000}, {'name':'Homer', 'age':39}]
from operator import itemgetter
newlist = sorted(list_to_be_sorted, key=itemgetter('age'))[-1]['name']
print(newlist)

