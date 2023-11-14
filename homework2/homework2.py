#!/usr/bin/env python
# coding: utf-8

# In[1]:


import random
from sklearn import linear_model
from matplotlib import pyplot as plt
from collections import defaultdict
import gzip


# In[2]:


def assertFloat(x):
    assert type(float(x)) == float

def assertFloatList(items, N):
    assert len(items) == N
    assert [type(float(x)) for x in items] == [float]*N


# In[3]:


answers = {}


# In[4]:


def parseData(fname):
    for l in open(fname):
        yield eval(l)


# In[5]:


import os


file_path = os.path.expanduser('~/Desktop/UCSD/0_study/CSE258/HW1/tast2_classification_beer preference/beer_50000.json')

data = list(parseData(file_path))


# In[6]:


random.seed(0)
random.shuffle(data)


# In[7]:


data[5]


# In[8]:


dataTrain = data[:25000]
dataValid = data[25000:37500]
dataTest = data[37500:]


# In[9]:


yTrain = [d['beer/ABV'] > 7 for d in dataTrain]
yValid = [d['beer/ABV'] > 7 for d in dataValid]
yTest = [d['beer/ABV'] > 7 for d in dataTest]


# In[ ]:





# In[10]:


categoryCounts = defaultdict(int)
for d in data:
    categoryCounts[d['beer/style']] += 1


# In[11]:


categories = [c for c in categoryCounts if categoryCounts[c] > 1000]


# In[12]:


catID = dict(zip(list(categories),range(len(categories))))


# In[13]:


max_length_train = max([len(d['review/text']) for d in dataTrain])
min_length_train = min([len(d['review/text']) for d in dataTrain])


# In[14]:


def feat(d, includeCat = True, includeReview = True, includeLength = True):
    global min_length_train, max_length_train
    
    featList = []
    
    if includeCat:
        # One-hot coding for beer style
        featList += [0.0] * len(catID)
        if d['beer/style'] in catID:
            featList[catID[d['beer/style']]] = 1
    
    if includeReview:
        #Add five ratings
        featList += [d['review/aroma'], d['review/taste'], d['review/appearance'], d['review/palate'], d['review/overall']]
        
    if includeLength:
        #Add review length and normalize it 
        normalized_length = (len(d['review/text']) - min_length_train) / (max_length_train - min_length_train)
        featList.append(normalized_length)
        featList = [1] + featList
    return featList


# In[15]:


def pipeline(reg, includeCat = True, includeReview = True, includeLength = True):
   
    XTrain = [feat(d, includeCat, includeReview, includeLength) for d in dataTrain]
    XValid = [feat(d, includeCat, includeReview, includeLength) for d in dataValid]
    XTest = [feat(d, includeCat, includeReview, includeLength) for d in dataTest]
    
    model = linear_model.LogisticRegression(C=reg, class_weight='balanced')
    model.fit(XTrain, yTrain)
    
    predValid = model.predict(XValid)
    predTest = model.predict(XTest)
    
    TP = sum([(p and l) for (p,l) in zip(predValid, yValid)])
    TN = sum([(not p and not l) for (p,l) in zip(predValid, yValid)])
    FP = sum([(p and not l) for (p,l) in zip(predValid, yValid)])
    FN = sum([(not p and l) for (p,l) in zip(predValid, yValid)])
    
    validBER = 1 - 0.5 * (TP / (TP + FN) + TN / (TN + FP))
    
    TP = sum([(p and l) for (p,l) in zip(predTest, yTest)])
    TN = sum([(not p and not l) for (p,l) in zip(predTest, yTest)])
    FP = sum([(p and not l) for (p,l) in zip(predTest, yTest)])
    FN = sum([(not p and l) for (p,l) in zip(predTest, yTest)])
    
    testBER = 1 - 0.5 * (TP / (TP + FN) + TN / (TN + FP))
    
    return model, validBER, testBER


# In[16]:


### Question 1


# In[17]:


mod, validBER, testBER = pipeline(10, True, False, False)


# In[18]:


answers['Q1'] = [validBER, testBER]


# In[19]:


assertFloatList(answers['Q1'], 2)


# In[20]:


print(answers)


# In[21]:


### Question 2


# In[22]:


mod, validBER, testBER = pipeline(10, includeCat=True, includeReview=True, includeLength=True)


# In[23]:


answers['Q2'] = [validBER, testBER]


# In[24]:


assertFloatList(answers['Q2'], 2)


# In[25]:


print(answers)


# In[26]:


### Question 3


# In[27]:


bestC = None
bestBER = float('inf')
for c in [0.001, 0.01, 0.1, 1, 10]:
    _, validBER, _ = pipeline(c, True, True, True)
    if validBER < bestBER:
        bestBER = validBER
        bestC = c


# In[28]:


mod, validBER, testBER = pipeline(bestC, True, True, True)


# In[29]:


answers['Q3'] = [bestC, validBER, testBER]


# In[30]:


assertFloatList(answers['Q3'], 3)


# In[31]:


print(answers)


# In[32]:


### Question 4


# In[33]:


mod, validBER, testBER_noCat = pipeline(1, False, True, True)


# In[34]:


mod, validBER, testBER_noReview = pipeline(1, True, False, True)


# In[35]:


mod, validBER, testBER_noLength = pipeline(1, True, True, False)


# In[36]:


answers['Q4'] = [testBER_noCat, testBER_noReview, testBER_noLength]


# In[37]:


assertFloatList(answers['Q4'], 3)


# In[38]:


print(answers)


# In[39]:


### Question 5


# In[42]:


import os


# In[43]:


path =os.path.expanduser( "~/Desktop/amazon_reviews_us_Musical_Instruments_v1_00.tsv.gz")
f = gzip.open(path, 'rt', encoding="utf8")

header = f.readline()
header = header.strip().split('\t')


# In[44]:


header


# In[45]:


dataset = []

pairsSeen = set()

for line in f:
    fields = line.strip().split('\t')
    d = dict(zip(header, fields))
    ui = (d['customer_id'], d['product_id'])
    if ui in pairsSeen:
        print("Skipping duplicate user/item:", ui)
        continue
    pairsSeen.add(ui)
    d['star_rating'] = int(d['star_rating'])
    d['helpful_votes'] = int(d['helpful_votes'])
    d['total_votes'] = int(d['total_votes'])
    dataset.append(d)


# In[46]:


dataTrain = dataset[:int(len(dataset)*0.9)]
dataTest = dataset[int(len(dataset)*0.9):]


# In[47]:


usersPerItem = defaultdict(set) # Maps an item to the users who rated it
itemsPerUser = defaultdict(set) # Maps a user to the items that they rated
itemNames = {}
ratingDict = {} # To retrieve a rating for a specific user/item pair
reviewsPerUser = defaultdict(list)
ratingsPerUserItem = defaultdict(float)

for d in dataTrain:
    user, item = d['customer_id'], d['product_id']
    usersPerItem[item].add(user)
    itemsPerUser[user].add(item)
    itemNames[item] = d['product_title']
    ratingsPerUserItem[(user, item)] = d['star_rating']
    
#for d in dataset:
    
    


# In[48]:


userAverages = {}
itemAverages = {}

for u in itemsPerUser:
    userAverages[u] = sum([ratingsPerUserItem[(u,i)] for i in itemsPerUser[u]]) / len(itemsPerUser[u])
    
for i in usersPerItem:
    itemAverages[i] = sum([ratingsPerUserItem[(u,i)] for u in usersPerItem[i]]) / len(usersPerItem[i])

ratingMean = sum([d['star_rating'] for d in dataTrain]) / len(dataTrain)


# In[49]:


def Jaccard(s1, s2):
    numer = len(s1.intersection(s2))
    denom = len(s1.union(s2))
    return numer/denom if denom != 0 else 0


# In[50]:


def mostSimilar(i, N):
    similarities = []
    users = usersPerItem[i]
    for i2 in usersPerItem:
        if i2 == i: continue
        sim = Jaccard(users, usersPerItem[i2])
        similarities.append((sim,i2))
    similarities.sort(reverse=True)
    return similarities[:N]


# In[51]:


query = 'B00KCHRKD6'


# In[52]:


ms = mostSimilar(query, 10)


# In[53]:


answers['Q5'] = ms


# In[54]:


assertFloatList([m[0] for m in ms], 10)


# In[55]:


print(answers)


# In[56]:


### Question 6


# In[57]:


def MSE(y, ypred):
    return sum([(a-b)**2 for a,b in zip(y,ypred)]) / len(y)


# In[58]:


def predictRating(user,item):
    if item not in itemAverages:
        return ratingMean

    ratingsSum = 0
    similaritiesSum = 0
    
    for j in itemsPerUser[user]:  # j is another item that the user has rated
        if j == item: continue  # We don't consider the item itself in the sum
        sim = Jaccard(usersPerItem[item], usersPerItem[j])  # Computing similarity between items i and j
        
        ratingsSum += (ratingsPerUserItem[(user, j)] - itemAverages[j]) * sim
        similaritiesSum += sim
        
    if similaritiesSum == 0:  # If no similar items exist, return the average rating of the item
        return itemAverages[item]
    
    predictedRating = itemAverages[item] + (ratingsSum / similaritiesSum)
    return predictedRating


# In[59]:


alwaysPredictMean = [ratingMean for d in dataTest]


# In[60]:


simPredictions = [predictRating(d['customer_id'], d['product_id']) for d in dataTest]


# In[61]:


labels = [d['star_rating'] for d in dataTest]


# In[ ]:





# In[62]:


answers['Q6'] = MSE(simPredictions, labels)


# In[63]:


assertFloat(answers['Q6'])


# In[64]:


print(answers)


# In[65]:


### Question 7


# In[68]:


reviewDatesPerUserItem = defaultdict(str)

for d in dataTrain:
    user, item = d['customer_id'], d['product_id']
    reviewDatesPerUserItem[(user, item)] = d['review_date']


# In[83]:


def timeWeightedPrediction(user, item, lambda_val=0.001):
    if item not in itemAverages:
        return ratingMean

    ratingsSum = 0
    similaritiesSum = 0
    
    for j in itemsPerUser[user]:
        if j == item: continue
        sim = Jaccard(usersPerItem[item], usersPerItem[j])
        
        # Check if the dates exist in the dictionary
        user_item_date = reviewDatesPerUserItem.get((user, item), None)
        user_j_date = reviewDatesPerUserItem.get((user, j), None)
        
        if not user_item_date or not user_j_date:
            continue  # skip this pair if any date is missing
        
        # Time-weighted decay
        time_difference = days_difference(user_item_date, user_j_date)
        decay = math.exp(-lambda_val * time_difference)
        
        ratingsSum += (ratingsPerUserItem[(user, j)] - itemAverages[j]) * sim * decay
        similaritiesSum += sim * decay

    if similaritiesSum == 0:
        return itemAverages[item]
    
    predictedRating = itemAverages[item] + (ratingsSum / similaritiesSum)
    return predictedRating


# In[84]:


timeWeightedPredictions = [timeWeightedPrediction(d['customer_id'], d['product_id']) for d in dataTest]


# In[85]:


# Computing MSE for the time-weighted predictions
itsMSE = MSE(timeWeightedPredictions, labels)


# In[86]:


answers['Q7'] = ["The design is a time-weighted collaborative filtering approach that considers the time difference between item reviews and other item reviews to emphasize the importance of recent ratings, providing a more accurate rating prediction. The Jaccard similarity is used to calculate the similarity between items, and exponential decay is employed to assign a time weight to each similarity (with a chosen lambda of 0.001). The predicted rating is based on the average rating of the item, the similarity between items, and the aforementioned time decay factor.", itsMSE]


# In[87]:


assertFloat(answers['Q7'][1])


# In[88]:


f = open("answers_hw2.txt", 'w')
f.write(str(answers) + '\n')
f.close()


# In[89]:


print(answers)


# In[ ]:




