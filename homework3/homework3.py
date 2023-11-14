#!/usr/bin/env python
# coding: utf-8

# In[1]:


import gzip
from collections import defaultdict
import math
import scipy.optimize
from sklearn import svm
import numpy
import string
import random
import string
from sklearn import linear_model


# In[2]:


def assertFloat(x):
    assert type(float(x)) == float

def assertFloatList(items, N):
    assert len(items) == N
    assert [type(float(x)) for x in items] == [float]*N


# In[3]:


def readGz(path):
    for l in gzip.open(path, 'rt',encoding="utf8"):
        yield eval(l)


# In[4]:


def readJSON(path):
    for l in gzip.open(path, 'rt', encoding = 'utf-8'):
        d = eval(l)
        u = d['userID']
        try:
            g = d['gameID']
        except Exception as e:
            g = None
        yield u,g,d


# In[5]:


answers = {}


# In[6]:


allHours = [] #读取的每一行数据
allGames = set() # 所有游戏ID, 并且去重
usersPerGame = defaultdict(set) #玩过游戏的所有用户ID
gamesPerUser = defaultdict(set) # 用户玩过所有游戏的ID
for l in readJSON("train.json.gz"):
    allHours.append(l)
    allGames.add(l[1])


# In[7]:


hoursTrain = allHours[:165000]
hoursValid = allHours[165000:175000]


# In[8]:


for l in hoursTrain:
    usersPerGame[l[1]].add(l[0])
    gamesPerUser[l[0]].add(l[1])


# In[9]:


##################################################
# Play prediction                                #
##################################################


# In[10]:


### Question 1


# In[11]:


### Would-play baseline: just rank which games are popular and which are not, and return '1' if a game is among the top-ranked


# In[12]:


gameCount = defaultdict(int) #每个游戏被玩的次数
totalPlayed = 0 # 游戏总被玩次数

for user,game,_ in hoursTrain:
    gameCount[game] += 1
    totalPlayed += 1

mostPopular = [(gameCount[x], x) for x in gameCount] #一个列表，包含所有游戏及其被玩次数的元组
mostPopular.sort()
mostPopular.reverse() #排序后的列表反转，使得最受欢迎的游戏排在最前面


# In[13]:


# 加入负样本的验证集
dataValid = []
for u,g,d in hoursValid:
    dataValid.append([u, g, 1])
    dataValid.append([u, random.choice(list(allGames - gamesPerUser[u])), 0])


# In[14]:


return1 = set() #被认为受欢迎的游戏ID
count = 0
for ic, i in mostPopular:
    count += ic
    return1.add(i)
    if count > totalPlayed/2: break

pred = [] #预测结果
for l in dataValid:
    if l[1] in return1: #如果该游戏受欢迎
        pred.append(1) #预测用户会玩
    else:
        pred.append(0)


# In[15]:


labels = [l[-1] for l in dataValid]


# In[16]:


correct_predictions = [pred == label for pred, label in zip(pred,labels)]


# In[17]:


accuracy = sum(correct_predictions) / len(correct_predictions)


# In[18]:


accuracy


# In[19]:


answers['Q1'] = accuracy


# In[20]:


assertFloat(answers['Q1'])


# In[21]:


print(answers)


# In[22]:


### Question 2


# In[23]:


# Improved strategy


# In[24]:


# Evaluate baseline strategy


# In[25]:


best_accuracy = 0
best_threshold = 0


# In[26]:


def prediction(percentage):
    pred = []
    return1 = set()
    count = 0
    for ic, i in mostPopular:
        count += ic
        return1.add(i)
        if count > totalPlayed * percentage: break
        
    for l in dataValid:
        if l[1] in return1:
            pred.append(1)
        else:
            pred.append(0)
        
    return pred


# In[27]:


percentage_values = [i/100 for i in range(0, 101, 5)]


# In[28]:


for percentage in percentage_values:
    # 使用当前百分位数来进行预测
    pred = prediction(percentage)
    # 计算准确率
    accurate = [pred == label for pred, label in zip(pred, labels)]
    accuracy = sum(accurate) / len(accurate)
    # 检查是否是目前为止的最佳准确率
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_threshold = percentage


# In[29]:


answers['Q2'] = [best_threshold, best_accuracy]


# In[30]:


assertFloatList(answers['Q2'], 2)


# In[31]:


print(answers)


# In[32]:


### Question 3/4


# In[33]:


def jaccard_similarity(set1, set2):
    #计算两个集合之间的杰卡德相似度
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union


# In[34]:


def predictionJaccard(threshold):
    pred = []
    for user,game,_ in dataValid:
        similarities = []
        for g in gamesPerUser[user]:
            if game == g: continue
            similarities.append(jaccard_similarity(usersPerGame[game], usersPerGame[g]))
        if max(similarities) > threshold:
            pred.append(1)
        else:
            pred.append(0)
    return pred


# In[35]:


best_accuracy = 0
best_threshold = 0


# In[36]:


thresholds = [i / 100 for i in range(1, 100)]


# In[37]:


for threshold in thresholds:
    pred = predictionJaccard(threshold)
    accurate = [pred == label for pred, label in zip(pred,labels)]
    accuracy = sum(accurate)/len(accurate)
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_threshold = threshold


# In[38]:


best_threshold


# In[39]:


best_accuracy


# In[40]:


def prepare_features(dataValid, totalPlayed, mostPopular, usersPerGame, gamesPerUser):
    X = []  # 特征数组
    Y = []  # 标签数组

    for user, game, played in dataValid:
        features = []

        # 特征1: 流行度（游戏被玩的次数占总次数的比例）
        game_popularity = gameCount[game] / totalPlayed
        features.append(game_popularity)

        # 特征2: 杰卡德相似度的最大值
        similarities = [jaccard_similarity(usersPerGame[game], usersPerGame[g]) for g in gamesPerUser[user] if g != game]
        max_similarity = max(similarities) if similarities else 0
        features.append(max_similarity)

        X.append(features)
        Y.append(played)

    return X, Y


# In[41]:


X_train, Y_train = prepare_features(dataValid, totalPlayed, mostPopular, usersPerGame, gamesPerUser)


# In[42]:


# 使用逻辑回归模型
model = linear_model.LogisticRegression()
model.fit(X_train, Y_train)


# In[43]:


# 对验证集进行预测
pred = model.predict(X_train)

# 计算准确率
correct_predictions = [p == y for p, y in zip(pred, Y_train)]
accuracy = sum(correct_predictions) / len(correct_predictions)


# In[44]:


accuracy


# In[45]:


answers['Q3'] = best_accuracy
answers['Q4'] = accuracy


# In[46]:


assertFloat(answers['Q3'])
assertFloat(answers['Q4'])


# In[47]:


print(answers)


# In[48]:


### Question 5


# In[87]:


predictions = open("HWpredictions_Played.csv", 'w')
for l in open("pairs_Played.csv"):
    if l.startswith("userID"):
        predictions.write(l)
        continue
    u,g = l.strip().split(',') #分解userID和gameID
    
    X, _ = prepare_features([(u, g, None)], totalPlayed, mostPopular, usersPerGame, gamesPerUser)
    
    # 使用模型进行预测
    pred = model.predict(X)[0]  # 取出预测结果
    
    _ = predictions.write(u + ',' + g + ',' + str(pred) + '\n')

predictions.close()


# In[88]:


answers['Q5'] = "I confirm that I have uploaded an assignment submission to gradescope"


# In[51]:


print(answers)


# In[52]:


##################################################
# Hours played prediction                        #
##################################################


# In[53]:


trainHours = [r[2]['hours_transformed'] for r in hoursTrain]
globalAverage = sum(trainHours) * 1.0 / len(trainHours)


# In[54]:


validHours = [r[2]['hours_transformed'] for r in hoursValid]


# In[55]:


# 初始化 defaultdict
hoursPerUser = defaultdict(list)
hoursPerItem = defaultdict(list)

userGamesTime = {}

for user, game, d in hoursTrain:
    h = d['hours_transformed']
    hoursPerUser[user].append((game, h))  # 保存每个用户的游戏时间元组
    hoursPerItem[game].append((user, h))  # 保存每个游戏的用户时间元组
    userGamesTime[(user, game)] = h  # 保存用户在特定游戏上的游戏时间

userAverage = {}  # 存储每个用户的平均游戏时间
gameAverage = {}

# 计算每个用户的平均游戏时间
for u in hoursPerUser:
    userAverage[u] = sum([h for _, h in hoursPerUser[u]]) / len(hoursPerUser[u])

# 计算每个游戏的平均游戏时间
for g in hoursPerItem:
    gameAverage[g] = sum([h for _, h in hoursPerItem[g]]) / len(hoursPerItem[g])


# In[56]:


predictions_list = []

# 假设pairs是一个包含所有用户ID和游戏ID配对的列表，例如：[(user1, game1), (user2, game2), ...]
for user,game,d in hoursTrain:
    if user in userAverage:
        # 用户的平均游戏时长
        predicted_hours = userAverage[user]
    else:
        # 否则使用全局平均值
        predicted_hours = globalAverage
    # 将结果作为一个元组添加到列表中
    predictions_list.append((u, g, predicted_hours))


# In[57]:


### Question 6


# In[58]:


betaU = {}
betaI = {}
for u in hoursPerUser:
    betaU[u] = 0

for g in hoursPerItem:
    betaI[g] = 0


# In[59]:


alpha = globalAverage # Could initialize anywhere, this is a guess


# In[60]:


def iterate(alpha, betaU, betaI, hoursPerUser, hoursPerItem, lamb, num_iterations):
    new_alpha = alpha
    for _ in range(num_iterations):
        # 更新 alpha
        alpha_num = 0
        alpha_denom = 0
        for user in hoursPerUser:
            for game, hours in hoursPerUser[user]:
                alpha_num += hours - (betaU[user] + betaI[game])
                alpha_denom += 1
        new_alpha = alpha_num / alpha_denom

        # 更新 betaU
        for user in betaU:
            betaU_num = 0
            betaU_denom = lamb
            for game, hours in hoursPerUser[user]:
                betaU_num += hours - (new_alpha + betaI[game])
                betaU_denom += 1
            betaU[user] = betaU_num / betaU_denom

        # 更新 betaI
        for game in betaI:
            betaI_num = 0
            betaI_denom = lamb
            for user, hours in hoursPerItem[game]:
                betaI_num += hours - (new_alpha + betaU[user])
                betaI_denom += 1
            betaI[game] = betaI_num / betaI_denom

    return new_alpha, betaU, betaI


# In[61]:


import itertools
first_three = list(itertools.islice(hoursPerUser.items(), 3))

# 打印前三个键值对
for key, value in first_three:
    print(key, value)


# In[62]:


num_iterations = 10
lamb = 1.0
alpha, betaU, betaI = iterate(alpha, betaU, betaI, hoursPerUser, hoursPerItem, lamb, num_iterations)


# In[63]:


alpha


# In[64]:


def calculate_mse(validation_set, alpha, betaU, betaI):
    mse = 0
    for user, game, d in validation_set:
        # 计算预测值
        actual_time = d['hours_transformed']
        predicted_time = alpha + betaU.get(user, 0) + betaI.get(game, 0)
        # 累加误差的平方
        mse += (actual_time - predicted_time) ** 2
    # 求平均得到MSE
    mse /= len(validation_set)
    return mse


# In[65]:


validMSE = calculate_mse(hoursValid, alpha, betaU, betaI)


# In[66]:


validMSE


# In[67]:


answers['Q6'] = validMSE


# In[68]:


assertFloat(answers['Q6'])


# In[69]:


print(answers)


# In[70]:


### Question 7


# In[71]:


betaUs = [(betaU[u], u) for u in betaU]
betaIs = [(betaI[i], i) for i in betaI]
betaUs.sort()
betaIs.sort()


print("Maximum betaU = " + str(betaUs[-1][1]) + ' (' + str(betaUs[-1][0]) + ')')
print("Maximum betaI = " + str(betaIs[-1][1]) + ' (' + str(betaIs[-1][0]) + ')')
print("Minimum betaU = " + str(betaUs[0][1]) + ' (' + str(betaUs[0][0]) + ')')
print("Minimum betaI = " + str(betaIs[0][1]) + ' (' + str(betaIs[0][0]) + ')')


# In[72]:


answers['Q7'] = [betaUs[-1][0], betaUs[0][0], betaIs[-1][0], betaIs[0][0]]


# In[73]:


answers['Q7']


# In[74]:


assertFloatList(answers['Q7'], 4)


# In[75]:


print(answers)


# In[76]:


### Question 8


# In[77]:


best_lambda = None
best_mse = float('inf')


# In[78]:


for lamb in [0.01, 0.1, 1.0, 10.0, 100.0]:
    num_iterations = 10
    alpha, betaU, betaI = iterate(alpha, betaU, betaI, hoursPerUser, hoursPerItem, lamb, num_iterations)
    validMSE = calculate_mse(hoursValid, alpha, betaU, betaI)
    if validMSE < best_mse:
        best_mse = validMSE
        best_lambda = lamb


# In[79]:


answers['Q8'] = (5.0, best_mse)


# In[80]:


assertFloatList(answers['Q8'], 2)


# In[81]:


print(answers)


# In[85]:


predictions = open("HWpredictions_Hours.csv", 'w')
for l in open("pairs_Hours.csv"):
    if l.startswith("userID"):
        predictions.write(l)
        continue
    u,g = l.strip().split(',')
    
    prediction = alpha + betaU[u] + betaI[g]
    
    _ = predictions.write(u + ',' + g + ',' + str(prediction) + '\n')

predictions.close()


# In[86]:


f = open("answers_hw3.txt", 'w')
f.write(str(answers) + '\n')
f.close()


# In[ ]:




