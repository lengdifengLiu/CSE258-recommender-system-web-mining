#!/usr/bin/env python
# coding: utf-8

# In[27]:


import json
from collections import defaultdict
from sklearn import linear_model
import sklearn
import numpy as np
import random
import gzip
import dateutil.parser
import math
import matplotlib.pyplot as plt


# In[28]:


answers = {}


# In[29]:


import os


# In[30]:


desktop_path = os.path.expanduser("~/Desktop/UCSD/0_study/CSE258/HW1/task1_regression")
print(desktop_path)
file_path = os.path.join(desktop_path, "fantasy_10000.json.gz")
print(file_path)


# In[31]:


def assertFloat(x):
    assert type(float(x)) == float

def assertFloatList(items, N):
    assert len(items) == N
    assert [type(float(x)) for x in items] == [float]*N


# # Question 1

# In[32]:


f = gzip.open(file_path)
dataset = []
for l in f:
    dataset.append(json.loads(l))


# In[33]:


dataset = [d for d in dataset if 'review_text' in d]
ratings = [d['rating'] for d in dataset]
lengths = [len(d['review_text']) for d in dataset]


# In[34]:


lengths = np.array(lengths)
scaled_lengths = (lengths - np.min(lengths)) / (np.max(lengths) - np.min(lengths))


# In[35]:


scaled_lengths = np.array(scaled_lengths)
X = np.column_stack((np.ones(len(scaled_lengths)), scaled_lengths))# Note the inclusion of the constant term
y = np.matrix(ratings).T
X = np.asarray(X)
y = np.asarray(y)


# In[36]:


model = sklearn.linear_model.LinearRegression(fit_intercept=False)
model.fit(X, y)


# In[37]:


theta = model.coef_
theta


# In[38]:


y_pred = model.predict(X)


# In[39]:


sse = sum([x**2 for x in (y - y_pred)])


# In[40]:


MSE = sse / len(y)
MSE


# In[41]:


answers['Q1'] = [theta[0][0], theta[0][1], MSE[0]]
answers['Q1']


# In[42]:


assertFloatList(answers['Q1'], 3)


# In[43]:


answers


# # Question 2

# In[44]:


import dateutil.parser


# In[45]:


t = [] 

for data in dataset:
    date_added = data.get('date_added') 
    if date_added:
        parsed_date = dateutil.parser.parse(date_added)  
        t.append(parsed_date)  


# In[46]:


weekdays = []
months = []

for date in t:
    weekday = date.weekday()
    month = date.month
    weekdays.append(weekday)
    months.append(month)
    


# In[47]:


weekday_vector_list = []

for i in range(len(weekdays)):
    weekday_vector = [0] * 6
    if weekdays[i]:
        weekday_vector[weekdays[i] - 1] = 1
    weekday_vector_list.append(weekday_vector)
        


# In[48]:


month_vector_list = []

for i in range(len(months)):
    month_vector = [0] * 11
    if months[i] >= 1:
        month_vector[months[i] - 2] = 1
    month_vector_list.append(month_vector)


# In[49]:


type(weekday_vector_list)


# In[50]:


type(weekdays)


# In[51]:


def feature_vec(length, month, weekday):
    feature = []
    for i in range(len(length)):
        feature.append([1]+[length[i]]+weekday[i]+month[i])
    return feature


# In[52]:


scaled_lengths


# In[53]:


feature_vector = feature_vec(scaled_lengths, month_vector_list, weekday_vector_list)


# In[54]:


answers['Q2'] = [feature_vector[0], feature_vector[1]]
answers['Q2']


# In[55]:


assertFloatList(answers['Q2'][0], 19)
assertFloatList(answers['Q2'][1], 19)


# In[56]:


print(answers)


# # Question 3

# In[57]:


weekdays_vec = list(map(lambda x: [x], weekdays))
months_vec = list(map(lambda x: [x], months))
X_3 = feature_vec(scaled_lengths, weekdays_vec, months_vec)


# In[58]:


X_3 = np.array(X_3)


# In[59]:


feature_vector = np.array(feature_vector) 


# In[60]:


model3_1 = sklearn.linear_model.LinearRegression(fit_intercept=False)
model3_1.fit(X_3, y)


# In[61]:


model3_2 = sklearn.linear_model.LinearRegression(fit_intercept=False)
model3_2.fit(feature_vector, y)


# In[62]:


y3_1_pred = model3_1.predict(X_3)


# In[63]:


y3_2_pred = model3_2.predict(feature_vector)


# In[64]:


sse3_1 = sum([i**2 for i in (y - y3_1_pred)])
MSE3_1 = sse3_1 / len(y)
MSE3_1 = float(MSE3_1)
MSE3_1


# In[65]:


sse3_2 = sum([i**2 for i in (y - y3_2_pred)])
MSE3_2 = sse3_2 / len(y)
MSE3_2 = float(MSE3_2)
MSE3_2


# In[66]:


answers['Q3'] = [MSE3_1, MSE3_2]


# In[67]:


assertFloatList(answers['Q3'], 2)


# In[68]:


print(answers)


# # Question 4

# In[69]:


random.seed(0)
random.shuffle(dataset)


# In[70]:


X_2 = feature_vector
X_2


# In[71]:


X_3


# In[72]:


Y = [d['rating'] for d in dataset]


# In[73]:


train2, test2 = X_2[:len(X_2)//2], X_2[len(X_2)//2:]
train3, test3 = X_3[:len(X_3)//2], X_3[len(X_3)//2:]
trainY, testY = Y[:len(Y)//2], Y[len(Y)//2:]


# In[74]:


model4_2 = sklearn.linear_model.LinearRegression(fit_intercept=False)
model4_2.fit(train2, trainY)


# In[75]:


model4_3 = sklearn.linear_model.LinearRegression(fit_intercept=False)
model4_3.fit(train3, trainY)


# In[76]:


y4_2_pred = model4_2.predict(test2)


# In[77]:


y4_3_pred = model4_3.predict(test3)


# In[78]:


sse4_2 = sum([i**2 for i in (testY - y4_2_pred)])
MSE4_2 = sse4_2 / len(testY)
MSE4_2


# In[79]:


sse4_3 = sum([i**2 for i in (testY - y4_3_pred)])
MSE4_3 = sse4_3 / len(testY)
MSE4_3


# In[80]:


answers['Q4'] = [MSE4_2, MSE4_3]
answers['Q4']


# In[81]:


assertFloatList(answers['Q4'], 2)


# In[82]:


print(answers)


# In[ ]:





# # Question5

# In[264]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import numpy as np
import os


# In[265]:


desktop_path_beer = os.path.expanduser("~/Desktop/UCSD/0_study/CSE258/HW1/tast2_classification_beer preference")
beer_50000_file_path = os.path.join(desktop_path_beer, "beer_50000.json")


# In[266]:


f = open(beer_50000_file_path)
data = []
for l in f:
    data.append(eval(l))


# In[267]:


X = np.array([len(d['review/text']) for d in data]).reshape(-1,1)


# In[268]:


y = np.array([d['review/overall'] >= 4  for d in data])


# In[269]:


X[:10]


# In[270]:


y[:10]


# In[271]:


model = LogisticRegression(class_weight='balanced')


# In[272]:


model.fit(X,y)


# In[273]:


predictions = model.predict(X)


# In[274]:


confusion = confusion_matrix(y, predictions)

TP = confusion[1, 1]
TN = confusion[0, 0]
FP = confusion[0, 1]
FN = confusion[1, 0]

FPR = FP / (FP + TN)
FNR = FN / (FN + TP)

BER = (FPR + FNR) / 2


# In[275]:


answers['Q5'] = [TP, TN, FP, FN, BER]


# In[276]:


assertFloatList(answers['Q5'], 5)


# In[277]:


print(answers)


# # Question6

# In[278]:


precs = []


# In[279]:


scores = model.decision_function(X)


# In[280]:


scores


# In[281]:


scoreslabels = list(zip(scores,y))


# In[282]:


scoreslabels.sort(reverse = True)


# In[283]:


scoreslabels


# In[284]:


sortedlabels = [x[1] for x in scoreslabels]


# In[285]:


sortedlabels


# In[286]:


for k in [1,100,1000,10000]:
    top_scores= sortedlabels[:k]
    TP_k = np.sum(top_scores)
    precision_k = TP_k / k
    precs.append(precision_k)


# In[287]:


precs


# In[288]:


answers['Q6'] = precs


# In[289]:


assertFloatList(answers['Q6'], 4)


# In[290]:


print(answers)


# # Question 7

# In[369]:


data[:3]


# In[370]:


beer_styles = np.array([d['beer/style'] for d in data]).reshape(-1,1)


# In[371]:


#X2 = np.array([d['review/taste'] for d in data]).reshape(-1,1)


# In[372]:


from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()


# In[373]:


encoded_styles = encoder.fit_transform(beer_styles)


# In[374]:


encoded_styles


# In[375]:


label_mapping = dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))


# In[376]:


print(label_mapping)


# In[377]:


X2_ = encoded_styles.tolist()


# In[378]:


type(X2_)


# In[379]:


X2_[0]


# In[380]:


X2_


# In[381]:


X2 = [[element] for element in X2_]


# In[382]:


X = X.tolist()


# In[383]:


type(X)


# In[384]:


X


# In[402]:


X_test = [X[i]+X2[i] for i in range(len(X))]


# In[405]:


len(X_test)


# In[406]:


X_test[:10]


# In[407]:


model_ = LogisticRegression(class_weight='balanced')


# In[408]:


model_.fit(X_test,y)


# In[410]:


y_pred = model_.predict(X_test)


# In[411]:


confusion = confusion_matrix(y, y_pred)

# 计算TP, TN, FP, FN
TP = confusion[1, 1]
TN = confusion[0, 0]
FP = confusion[0, 1]
FN = confusion[1, 0]

# 计算FPR和FNR
FPR = FP / (FP + TN)
FNR = FN / (FN + TP)

its_test_BER = (FPR + FNR) / 2


# In[412]:


its_test_BER


# In[413]:


its_test_BER < BER


# In[ ]:





# In[417]:


answers['Q7'] = ["I chose to use beer style as an additional feature and combine review length and beer style to form a new feature. Since beer style is a string type, I combine non-numeric features (beer style) is converted to numerical features, and I use the LabelEncoder in the scikit-learn library to implement it", its_test_BER]


# In[418]:


f = open("answers_hw1.txt", 'w')
f.write(str(answers) + '\n')
f.close()


# In[419]:


print(answers)


# In[ ]:




