#!/usr/bin/env python
# coding: utf-8

# In[1]:


import json
import gzip
import math
from collections import defaultdict
import numpy as np
from sklearn import linear_model
import random
import statistics


# In[2]:


def assertFloat(x):
    assert type(float(x)) == float

def assertFloatList(items, N):
    assert len(items) == N
    assert [type(float(x)) for x in items] == [float]*N


# In[3]:


answers = {}


# In[4]:


z = gzip.open("train.json.gz")

dataset = []
for l in z:
    d = eval(l)
    dataset.append(d)

z.close()


# In[5]:


dataset[:3]


# In[6]:


### Question 1


# In[7]:


def MSE(y, ypred):
    return np.mean((np.array(y) - np.array(ypred))**2)

def MAE(y, ypred):
    return np.mean(np.abs(np.array(y) - np.array(ypred)))


# In[8]:


reviewsPerUser = defaultdict(list) #a list of all reviews written by that user
reviewsPerItem = defaultdict(list) #a list of all reviews written for that game

for d in dataset:
    u,i = d['userID'],d['gameID']
    reviewsPerUser[u].append(d)
    reviewsPerItem[i].append(d)
    
for u in reviewsPerUser:
    reviewsPerUser[u].sort(key=lambda x: x['date'])
    
for i in reviewsPerItem:
    reviewsPerItem[i].sort(key=lambda x: x['date'])


# In[9]:


def feat1(d):
    return [d['hours']]


# In[10]:


X = [feat1(d) for d in dataset]
y = [len(d['text']) for d in dataset]


# In[11]:


mod =linear_model.LinearRegression()
mod.fit(X,y)

predictions = mod.predict(X)


# In[12]:


theta_1 = mod.coef_[0]


# In[13]:


mse_q1 = MSE(y, predictions)


# In[14]:


answers['Q1'] = [theta_1, mse_q1]
assertFloatList(answers['Q1'], 2)


# In[15]:


print(answers)


# In[16]:


### Question 2


# In[17]:


hours_list = [d['hours'] for d in dataset]
median_hours = np.median(hours_list)


# In[18]:


def feat2(d):
    hours = d['hours']
    hours_transformed = d['hours_transformed']
    sqrt_hours = np.sqrt(hours)     # square root transform
    hours_indicator = 1 if hours > median_hours else 0  # binary indicator
    return [1, hours, hours_transformed, sqrt_hours, hours_indicator]


# In[19]:


X = [feat2(d) for d in dataset]


# In[20]:


mod = linear_model.LinearRegression(fit_intercept=False)
mod.fit(X,y)
predictions = mod.predict(X)


# In[21]:


mse_q2 = np.mean((y - predictions) ** 2)


# In[22]:


answers['Q2'] = mse_q2


# In[23]:


assertFloat(answers['Q2'])


# In[24]:


print(answers)


# In[25]:


### Question 3


# In[26]:


def feat3(d):
    h = d['hours']  
    return [1, h > 1, h > 5, h > 10, h > 100, h > 1000]


# In[27]:


X = [feat3(d) for d in dataset]


# In[28]:


mod = linear_model.LinearRegression(fit_intercept=False)
mod.fit(X,y)
predictions = mod.predict(X)


# In[29]:


mse_q3 = MSE(y, predictions)


# In[30]:


answers['Q3'] = mse_q3


# In[31]:


assertFloat(answers['Q3'])


# In[32]:


print(answers)


# In[33]:


### Question 4


# In[34]:


def feat4(d):
    review_length = len(d['text'])
    return [1, review_length]


# In[35]:


X = [feat4(d) for d in dataset]
y = [d['hours'] for d in dataset]


# In[36]:


mod = linear_model.LinearRegression(fit_intercept=False)
mod.fit(X,y)
predictions = mod.predict(X)


# In[37]:


mse = MSE(y, predictions)
mae = MAE(y, predictions)


# In[38]:


explanation = "In my point of view, MSE can be better. Because MSE can better reflect the influence of outliers, it can be used to evaluate the model."


# In[39]:


answers['Q4'] = [mse, mae, explanation]


# In[40]:


assertFloatList(answers['Q4'][:2], 2)


# In[41]:


print(answers)


# In[42]:


### Question 5


# In[43]:


y_trans = np.array([d['hours_transformed'] for d in dataset])


# In[44]:


X = np.array([feat4(d) for d in dataset])


# In[45]:


mod = linear_model.LinearRegression(fit_intercept=False)
mod.fit(X,y_trans)
predictions_trans = mod.predict(X)


# In[46]:


mse_trans = MSE(y_trans, predictions_trans)


# In[47]:


predictions_untrans = 2**predictions_trans - 1


# In[48]:


y_original = 2**y_trans - 1


# In[49]:


mse_untrans = MSE(y_original, predictions_untrans)


# In[50]:


answers['Q5'] = [mse_trans, mse_untrans]


# In[51]:


assertFloatList(answers['Q5'], 2)


# In[52]:


print(answers)


# In[53]:


### Question 6


# In[54]:


def feat6(d):
    # Your one-hot encoding implementation
    one_hot_vector = [0] * 101
    hours_played = int(d['hours'])
    index = min(hours_played,99)
    one_hot_vector[0] = 1
    one_hot_vector[index+1] = 1
    return  one_hot_vector


# In[55]:


X = np.array([feat6(d) for d in dataset])
y = np.array([len(d['text']) for d in dataset])


# In[56]:


Xtrain, Xvalid, Xtest = X[:len(X)//2], X[len(X)//2:(3*len(X))//4], X[(3*len(X))//4:]
ytrain, yvalid, ytest = y[:len(X)//2], y[len(X)//2:(3*len(X))//4], y[(3*len(X))//4:]


# In[57]:


models = {}
mses = {}
bestC = None
bestMSE = float('inf')

for c in [1, 10, 100, 1000, 10000]:
    model = linear_model.Ridge(alpha=c,fit_intercept = False)
    model.fit(Xtrain, ytrain)
    # Predict on validation set
    ypred_valid = model.predict(Xvalid)
    mse_valid = MSE(yvalid, ypred_valid)
    
    # Store model and MSE
    models[c] = model
    mses[c] = mse_valid
    
    if mse_valid < bestMSE:
        bestC = c
        bestMSE = mse_valid


# In[58]:


best_model = models[bestC]


# In[59]:


predictions_test = best_model.predict(Xtest)
mse_test = MSE(ytest, predictions_test)


# In[60]:


mse_valid = bestMSE


# In[61]:


mse_test = MSE(ytest, predictions_test)


# In[62]:


answers['Q6'] = [bestC, mse_valid, mse_test]


# In[63]:


assertFloatList(answers['Q6'], 3)


# In[64]:


print(answers)


# In[65]:


### Question 7


# In[66]:


times = [d['hours_transformed'] for d in dataset]
median = statistics.median(times)


# In[67]:


notPlayed = [d for d in dataset if d['hours_transformed'] < 1]
nNotPlayed = len(notPlayed)


# In[68]:


answers['Q7'] = [median, nNotPlayed]


# In[69]:


assertFloatList(answers['Q7'], 2)


# In[70]:


print(answers)


# In[71]:


### Question 8


# In[72]:


def feat8(d):
    return [len(d['text'])]


# In[73]:


X = [feat8(d) for d in dataset]
y = [d['hours_transformed'] > median for d in dataset]


# In[74]:


mod = linear_model.LogisticRegression(class_weight='balanced')
mod.fit(X,y)
predictions = mod.predict(X) # Binary vector of predictions


# In[75]:


def rates(predictions, y):
    TP = sum(p and t for p, t in zip(predictions, y))
    TN = sum(not p and not t for p, t in zip(predictions, y))
    FP = sum(p and not t for p, t in zip(predictions, y))
    FN = sum(not p and t for p, t in zip(predictions, y))
    return TP, TN, FP, FN


# In[76]:


TP, TN, FP, FN = rates(predictions, y)


# In[77]:


BER = 1/2 * (FP / (FP + TN) + FN / (TP + FN)) 


# In[78]:


answers['Q8'] = [TP, TN, FP, FN, BER]


# In[79]:


assertFloatList(answers['Q8'], 5)


# In[80]:


print(answers)


# In[81]:


### Question 9


# In[82]:


from sklearn.metrics import precision_score


# In[83]:


probabilities = mod.predict_proba(X)[:, 1] 


# In[84]:


sortedByConfidence = list(zip(probabilities, y))
sortedByConfidence.sort(reverse=True)


# In[85]:


precs = []
for k in [5, 10, 100, 1000]:
# Determine the k-th score (0-indexed)
    threshold_score = sortedByConfidence[k - 1][0]
    
    effective_k = len([score for score, _ in sortedByConfidence if score >= threshold_score])
    
    # Calculate precision at effective k
    true_positive_count = sum(1 for _, label in sortedByConfidence[:effective_k] if label == 1)
    precision_at_k = true_positive_count / effective_k
    precs.append(precision_at_k)


# In[86]:


answers['Q9'] = precs


# In[87]:


assertFloatList(answers['Q9'], 4)


# In[88]:


print(answers)


# In[89]:


### Question 10


# In[90]:


y_trans = [d['hours_transformed'] for d in dataset]


# In[91]:


mod = linear_model.LinearRegression(fit_intercept=False)
mod.fit(X,y_trans)
predictions_trans = mod.predict(X)


# In[92]:


your_threshold = 0
best_BER =  0.472506390561468


# In[93]:


for threshold in np.arange(0.1, 10, 0.1):
    predictions_thresh = predictions_trans > threshold # This creates a boolean array
    TP, TN, FP, FN = rates(predictions_thresh, y)
    BER = 0.5 * ((FP / (FP + TN)) + (FN / (TP + FN)))
    
    if BER < best_BER:
        best_BER = BER
        your_threshold = threshold


# In[94]:


best_BER


# In[95]:


your_threshold


# In[96]:


answers['Q10'] = [your_threshold, best_BER]
assertFloatList(answers['Q10'], 2)


# In[97]:


print(answers)


# In[98]:


### Question 11


# In[99]:


dataTrain = dataset[:int(len(dataset)*0.9)]
dataTest = dataset[int(len(dataset)*0.9):]


# In[100]:


userHours = defaultdict(list)
itemHours = defaultdict(list)


# In[101]:


for d in dataTrain:
    user, item, hours = d['userID'], d['gameID'], d['hours']
    userHours[user].append(hours)
    itemHours[item].append(hours)


# In[102]:


userMedian = {user: statistics.median(times) for user, times in userHours.items()}
itemMedian = {item: statistics.median(times) for item, times in itemHours.items()}


# In[103]:


answers['Q11'] = [itemMedian['g35322304'], userMedian['u55351001']]


# In[104]:


assertFloatList(answers['Q11'], 2)


# In[105]:


print(answers)


# In[106]:


### Question 12


# In[107]:


allHours = [d['hours'] for d in dataTrain] 
globalMedian = statistics.median(allHours)


# In[108]:


def f12(u,i):
    if i in itemMedian and itemMedian[i] > globalMedian:
        return 1
    # 检查用户u玩的时间是否超过全局中位数，如果是，返回1
    elif i not in itemMedian and userMedian[u] > globalMedian:
        return 1
    # 如果上述条件都不符合，则返回0
    else:
        return 0


# In[109]:


preds = [f12(d['userID'], d['gameID']) for d in dataTest]


# In[110]:


y = [(d['hours'] > globalMedian) for d in dataTest] 


# In[111]:


accuracy = sum(pred == y_true for pred, y_true in zip(preds, y)) / len(y)


# In[112]:


answers['Q12'] = accuracy


# In[113]:


assertFloat(answers['Q12'])


# In[114]:


print(answers)


# In[115]:


### Question 13


# In[116]:


usersPerItem = defaultdict(set) # Maps an item to the users who rated it
itemsPerUser = defaultdict(set) # Maps a user to the items that they rated
itemNames = {}

for d in dataset:
    user,item = d['userID'], d['gameID']
    usersPerItem[item].add(user)
    itemsPerUser[user].add(item)


# In[117]:


def Jaccard(s1, s2):
    numer = len(s1.intersection(s2))
    denom = len(s1.union(s2))
    return numer / denom if denom else 0  


# In[118]:


def mostSimilar(i, func, N):
    similarities = []
    users = usersPerItem[i]
    for ii in usersPerItem:
        if ii == i: continue  # 不和自己比较
        sim = func(users, usersPerItem[ii])
        similarities.append((sim, ii))
    similarities.sort(reverse=True)  # 降序排序
    return similarities[:N]  # 返回前N个最相似的


# In[119]:


ms = mostSimilar(dataset[0]['gameID'], Jaccard, 10)


# In[120]:


answers['Q13'] = [ms[0][0], ms[-1][0]]
assertFloatList(answers['Q13'], 2)


# In[121]:


print(answers)


# In[122]:


### Question 14


# In[123]:


def mostSimilar14(i, func, N):
    similarities = []
    for ii in usersPerItem:
        if ii == i: continue  # 不与自身比较
        sim = func(i, ii)
        similarities.append((sim, ii))
    similarities.sort(reverse=True)  # 降序排序
    return similarities[:N]  # 返回前N个最相似的


# In[124]:


ratingDict = {}

for d in dataset:
    u,i = d['userID'], d['gameID']
    lab = 1 if d['hours'] > median_hours else -1
    ratingDict[(u,i)] = lab


# In[125]:


def Cosine(i1, i2):
    inter = usersPerItem[i1].intersection(usersPerItem[i2])
    numer = 0
    denom1 = 0
    denom2 = 0
    for u in inter:
        numer += ratingDict[(u,i1)]*ratingDict[(u,i2)]
    for u in usersPerItem[i1]:
        denom1 += ratingDict[(u,i1)]**2
    for u in usersPerItem[i2]:
        denom2 += ratingDict[(u,i2)]**2
    denom = math.sqrt(denom1) * math.sqrt(denom2)
    if denom == 0: return 0
    return numer / denom


# In[126]:


ms = mostSimilar14(dataset[0]['gameID'], Cosine, 10)


# In[127]:


answers['Q14'] = [ms[0][0], ms[-1][0]]


# In[128]:


assertFloatList(answers['Q14'], 2)


# In[129]:


print(answers)


# In[130]:


### Question 15


# In[131]:


ratingDict = {}

for d in dataset:
    u,i = d['userID'], d['gameID']
    lab = d['hours_transformed']
    ratingDict[(u,i)] = lab


# In[132]:


ms = mostSimilar14(dataset[0]['gameID'], Cosine, 10)


# In[133]:


answers['Q15'] = [ms[0][0], ms[-1][0]]


# In[134]:


assertFloatList(answers['Q15'], 2)


# In[135]:


f = open("answers_midterm.txt", 'w')
f.write(str(answers) + '\n')
f.close()


# In[136]:


print(answers)

