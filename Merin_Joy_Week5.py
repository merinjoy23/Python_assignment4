#!/usr/bin/env python
# coding: utf-8

# In[78]:


import sqlite3
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


# In[145]:


conn = sqlite3.connect('/Users/Merin/Desktop/HAP880/Assignment5/testClaims_hu.db')
df = pd.read_sql('select * from highUtilizationPredictionV2wco', conn)


# In[146]:


race = pd.get_dummies(df['race'], dummy_na=False)
df=pd.concat([df,race], axis=1)


# In[319]:


from sklearn.model_selection import train_test_split
tr, ts = train_test_split(df, test_size=0.2)


# In[320]:


cls = list(df.columns)
cls.remove('index')
cls.remove('race')
cls.remove('patient_id')
cls.remove('claimCount')
cls.remove('HighUtilizationY2')


# In[149]:


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import auc
from sklearn.metrics import roc_curve


# #### Building the logistic model

# In[260]:


lr = LogisticRegression(solver='lbfgs', C=0.9, max_iter=1000)
lr.fit(tr[cls], tr['HighUtilizationY2'])


# #### Generate AUC

# In[85]:


probs_lr = lr.predict_proba(ts[cls])[:,1]
fpr_lr, tpr_lr, thresholds_lr = roc_curve(ts['HighUtilizationY2'], probs_lr)
auc_lr=auc(fpr_lr, tpr_lr)
auc_lr


# #### Plot AUC

# In[86]:


plt.scatter(fpr_lr, tpr_lr)


# #### Building the Random Forest model

# In[87]:


rf = RandomForestClassifier(n_estimators=400)
rf.fit(tr[cls], tr['HighUtilizationY2'])


# #### Generate AUC

# In[88]:


probs_rf = rf.predict_proba(ts[cls])[:,1]
fpr_rf, tpr_rf, thresholds_rf = roc_curve(ts['HighUtilizationY2'], probs_rf)
auc_rf = auc(fpr_rf, tpr_rf)
auc_rf


# #### Plot AUC

# In[89]:


plt.scatter(fpr_rf, tpr_rf)


# #### AUC for Logistic and Random forest model

# In[90]:


plt.scatter(fpr_lr, tpr_lr)
plt.scatter(fpr_rf, tpr_rf)


# #### Classification threshold for recall 0.8 

# In[91]:


# Random Forest
r = 0
while tpr_rf[r] < 0.8:
    r = r+1
r


# In[92]:


tpr_rf[r], fpr_rf[r], thresholds_rf[r]


# In[93]:


# Logistic Regression
l=0
while tpr_lr[l] < 0.8:
    l = l+1
l


# In[94]:


tpr_lr[l], fpr_lr[l], thresholds_lr[l]


# In[95]:


# Percentage of positive class
ts['HighUtilizationY2'].mean()*100


# In[96]:


# Classification example if threshold is 0.05
probs_rf > 0.05


# #### Generating a synthetic patient with all 0s and age 65 (healthy patient)

# In[97]:


age = 65
ELIX = [0 for i in range(29)]
G = [0 for i in range(22)]
drugs = [0 for i in range(12)]
A = 0
AmN = 0
B = 0
H = 0
O = 0
U = 0
W = 0

d = [age] + ELIX + G + drugs + [A, AmN, B, H, O, U, W]


# In[98]:


dat = [d]


# #### Model predicts that the patient is not a high utilizer

# In[99]:


lr.predict_proba(dat)


# In[100]:


rf.predict_proba(dat)


# #### Generating a random patient

# In[101]:


# age
age = 65

# randomly select ELIX codes
ELIX = [np.random.randint(2) for i in range(29)]

#randomly select procedures
G = [np.random.randint(2) for i in range(22)]

#randomly select drug counts
drugs = [np.random.randint(13) for i in range(12)]

# zero all races
A = 0
AmN = 0
B = 0
H = 0
O = 0
U = 0
W = 0
# and randomly select race
r = np.random.randint(7)
if r == 0:
    A = 1
if r == 1:
    AmN = 1
if r == 2:
    B = 1
if r == 3:
    H = 1
if r == 4:
    O = 1
if r == 5:
    U = 1
if r == 6:
    W = 1  
d = [age] + ELIX + G + drugs + [A, AmN, B, H, O, U, W]


# In[102]:


dat = [d]


# #### Model says the patient is a high utilizer

# In[103]:


rf.predict_proba(dat)


# #### Simulating 1000 random patients of age 65

# In[104]:


lr_res = []
rf_res = []
for i in range(1000):
    age = 65
    ELIX = [np.random.randint(2) for i in range(29)]
    G = [np.random.randint(2) for i in range(22)]
    drugs = [np.random.randint(30) for i in range(12)] 
    A = 0
    AmN = 0
    B = 0
    H = 0
    O = 0
    U = 0
    W = 0
    r = np.random.randint(7)
    if r == 0:
        A = 1
    if r == 1:
        AmN = 1
    if r == 2:
        B = 1
    if r == 3:
        H = 1
    if r == 4:
        O = 1
    if r == 5:
        U = 1
    if r == 6:
        W = 1    
    d = [age] + ELIX + G + drugs + [A, AmN, B, H, O, U, W]
    dat = [d]
    lr_res.append(lr.predict_proba(dat)[:,1][0])
    rf_res.append(rf.predict_proba(dat)[:,1][0])


# #### Average and Standard deviation (sick patients are generated)

# In[105]:


np.mean(lr_res), np.std(lr_res)


# In[106]:


np.mean(rf_res), np.std(rf_res)


# #### Min and Max age

# In[107]:


maxa = tr['age'].max()
mina = tr['age'].min()


# #### Generate random data for all ages

# In[108]:


lr_ages = []
rf_ages = []
for age in range(mina, maxa+1):
    lr_res = []
    rf_res = []
    for i in range(1000):
        #age = 65
        ELIX = [np.random.randint(2) for i in range(29)]
        G = [np.random.randint(2) for i in range(22)]
        drugs = [np.random.randint(30) for i in range(12)] 
        A = 0
        AmN = 0
        B = 0
        H = 0
        O = 0
        U = 0
        W = 0
        r = np.random.randint(7)
        if r == 0:
            A = 1
        if r == 1:
            AmN = 1
        if r == 2:
            B = 1
        if r == 3:
            H = 1
        if r == 4:
            O = 1
        if r == 5:
            U = 1
        if r == 6:
            W = 1    
        d = [age] + ELIX + G + drugs + [A, AmN, B, H, O, U, W]
        dat = [d]
        lr_res.append(lr.predict_proba(dat)[:,1][0])
        rf_res.append(rf.predict_proba(dat)[:,1][0])
    lr_ages.append(sum(lr_res)/len(lr_res))
    rf_ages.append(sum(rf_res)/len(rf_res))


# #### Probability on random data

# In[109]:


plt.plot(range(mina, maxa+1), lr_ages)
plt.plot(range(mina, maxa+1), rf_ages)


# #### Running the model on training data

# In[110]:


probs_l = lr.predict_proba(tr[cls])[:,1]
probs_r = rf.predict_proba(tr[cls])[:,1]

tr_res = pd.DataFrame()
tr_res['age'] = tr['age']
tr_res['lr'] = probs_l
tr_res['rf'] = probs_r
gr=tr_res.groupby('age')

# This plot shows that old people are simply more sick
plt.plot(range(mina, maxa+1),list(gr['lr'].mean()))
plt.plot(range(mina, maxa+1),list(gr['rf'].mean()))


# #### Running the model on test data

# In[111]:


probs_l_t = lr.predict_proba(ts[cls])[:,1]
probs_r_t = rf.predict_proba(ts[cls])[:,1]

ts_res = pd.DataFrame()
ts_res['age'] = ts['age']
ts_res['lr'] = probs_l_t
ts_res['rf'] = probs_r_t
grts=ts_res.groupby('age')

# both train and test data
plt.plot(range(mina, maxa+1),list(gr['lr'].mean()), color='r')
plt.plot(range(mina, maxa+1),list(gr['rf'].mean()), color='b')
plt.plot(range(mina, maxa+1),list(grts['lr'].mean()), color='g')
plt.plot(range(mina, maxa+1),list(grts['rf'].mean()), color='y')


# #### Counts of people based on age

# In[112]:


plt.plot(range(mina, maxa+1),list(gr['lr'].count()), color='r')
plt.plot(range(mina, maxa+1),list(gr['rf'].count()), color='b')
plt.plot(range(mina, maxa+1),list(grts['lr'].count()), color='g')
plt.plot(range(mina, maxa+1),list(grts['rf'].count()), color='y')


# #### Smoothing the bumps from training data using a sliding window

# In[113]:


lr_ages = []
rf_ages = []
for age in range(mina, maxa+1):
    lr_res = []
    rf_res = []
    mn = max(age-5, mina)
    mx = min(age+5, maxa)
    dat = tr[(tr['age'] >= mn) & (tr['age'] <= mx)][cls]
    dat['age'] = age
    probs_l = lr.predict_proba(dat)[:,1]
    probs_r = rf.predict_proba(dat)[:,1]
       
    lr_ages.append(np.mean(probs_l))
    rf_ages.append(np.mean(probs_r))


# In[114]:


plt.plot(range(mina, maxa+1), lr_ages)
plt.plot(range(mina, maxa+1), rf_ages)


# #### Smoothing the bumps from test data using a sliding window

# In[115]:


lr_ages = []
rf_ages = []
lr_ages_s = []
rf_ages_s = []
for age in range(mina, maxa+1):
    lr_res = []
    rf_res = []
    mn = max(age-5, mina)
    mx = min(age+5, maxa)
    dat = ts[(ts['age'] >= mn) & (ts['age'] <= mx)][cls]
    dat['age'] = age
    probs_l = lr.predict_proba(dat)[:,1]
    probs_r = rf.predict_proba(dat)[:,1]
       
    lr_ages.append(np.mean(probs_l))
    rf_ages.append(np.mean(probs_r))
    lr_ages_s.append(np.std(probs_l))
    rf_ages_s.append(np.std(probs_r))


# In[116]:


plt.plot(range(mina, maxa+1), lr_ages)
plt.plot(range(mina, maxa+1), rf_ages)


# In[117]:


np.array(lr_ages) + np.array(lr_ages_s)


# #### Range of means and standard deviation

# In[118]:


plt.plot(range(mina, maxa+1), np.array(lr_ages) + np.array(lr_ages_s), color='r')
plt.plot(range(mina, maxa+1), np.array(lr_ages) - np.array(lr_ages_s), color='r')
plt.fill_between(range(mina, maxa+1), np.array(lr_ages) + np.array(lr_ages_s), np.array(lr_ages) - np.array(lr_ages_s), color='r')


# In[119]:


plt.ylim(-0.3,1.0)
plt.plot(range(mina, maxa+1), np.array(rf_ages) + np.array(rf_ages_s), color='b')
plt.plot(range(mina, maxa+1), np.array(rf_ages) - np.array(rf_ages_s), color='b')
plt.fill_between(range(mina, maxa+1), np.array(rf_ages) + np.array(rf_ages_s), np.array(rf_ages) - np.array(rf_ages_s), color='b')
plt.plot(range(mina, maxa+1), np.array(lr_ages) + np.array(lr_ages_s), color='r')
plt.plot(range(mina, maxa+1), np.array(lr_ages) - np.array(lr_ages_s), color='r')
plt.fill_between(range(mina, maxa+1), np.array(lr_ages) + np.array(lr_ages_s), np.array(lr_ages) - np.array(lr_ages_s), color='r')


# #### Looking at predicted values (average probabilities) per race

# In[120]:


tr_res['race'] = tr['race']
ts_res['race'] = ts['race']
gr_r_tr = tr_res.groupby('race')
gr_r_ts = ts_res.groupby('race')

p1 = gr_r_tr.lr.mean()
p2 = gr_r_tr.rf.mean()

# Logistic Regression
plt.bar(range(len(p1.index)), p1.values)
plt.xticks(range(len(p1.index)), p1.index)
plt.show()


# In[121]:


# Random Forest
plt.bar(range(len(p2.index)), p2.values)
plt.xticks(range(len(p2.index)), p2.index)
plt.show()


# #### Generate random data for all races

# In[46]:


lr_races = []
rf_races = []
lr_res = []
rf_res = []
for i in range(100):
        age = 65+np.random.randint(30)
        ELIX = [np.random.randint(2) for i in range(29)]
        G = [np.random.randint(2) for i in range(22)]
        drugs = [np.random.randint(30) for i in range(12)] 
        A = 1
        AmN = 0
        B = 0
        H = 0
        O = 0
        U = 0
        W = 0
        
        d = [age] + ELIX + G + drugs + [A, AmN, B, H, O, U, W]
        dat = [d]
        lr_res.append(lr.predict_proba(dat)[:,1][0])
        rf_res.append(rf.predict_proba(dat)[:,1][0])
  
lr_races.append(sum(lr_res)/len(lr_res))
rf_races.append(sum(rf_res)/len(rf_res))

lr_res = []
rf_res = []
for i in range(100):
        age = 65+np.random.randint(30)
        ELIX = [np.random.randint(2) for i in range(29)]
        G = [np.random.randint(2) for i in range(22)]
        drugs = [np.random.randint(30) for i in range(12)] 
        A = 0
        AmN = 1
        B = 0
        H = 0
        O = 0
        U = 0
        W = 0
        
        d = [age] + ELIX + G + drugs + [A, AmN, B, H, O, U, W]
        dat = [d]
        lr_res.append(lr.predict_proba(dat)[:,1][0])
        rf_res.append(rf.predict_proba(dat)[:,1][0])
  
lr_races.append(sum(lr_res)/len(lr_res))
rf_races.append(sum(rf_res)/len(rf_res))

lr_res = []
rf_res = []
for i in range(100):
        age = 65+np.random.randint(30)
        ELIX = [np.random.randint(2) for i in range(29)]
        G = [np.random.randint(2) for i in range(22)]
        drugs = [np.random.randint(30) for i in range(12)] 
        A = 0
        AmN = 0
        B = 1
        H = 0
        O = 0
        U = 0
        W = 0
        
        d = [age] + ELIX + G + drugs + [A, AmN, B, H, O, U, W]
        dat = [d]
        lr_res.append(lr.predict_proba(dat)[:,1][0])
        rf_res.append(rf.predict_proba(dat)[:,1][0])
  
lr_races.append(sum(lr_res)/len(lr_res))
rf_races.append(sum(rf_res)/len(rf_res))

lr_res = []
rf_res = []
for i in range(100):
        age = 65+np.random.randint(30)
        ELIX = [np.random.randint(2) for i in range(29)]
        G = [np.random.randint(2) for i in range(22)]
        drugs = [np.random.randint(30) for i in range(12)] 
        A = 0
        AmN = 0
        B = 0
        H = 1
        O = 0
        U = 0
        W = 0
        
        d = [age] + ELIX + G + drugs + [A, AmN, B, H, O, U, W]
        dat = [d]
        lr_res.append(lr.predict_proba(dat)[:,1][0])
        rf_res.append(rf.predict_proba(dat)[:,1][0])
  
lr_races.append(sum(lr_res)/len(lr_res))
rf_races.append(sum(rf_res)/len(rf_res))

lr_res = []
rf_res = []
for i in range(100):
        age = 65+np.random.randint(30)
        ELIX = [np.random.randint(2) for i in range(29)]
        G = [np.random.randint(2) for i in range(22)]
        drugs = [np.random.randint(30) for i in range(12)] 
        A = 0
        AmN = 0
        B = 0
        H = 0
        O = 1
        U = 0
        W = 0
        
        d = [age] + ELIX + G + drugs + [A, AmN, B, H, O, U, W]
        dat = [d]
        lr_res.append(lr.predict_proba(dat)[:,1][0])
        rf_res.append(rf.predict_proba(dat)[:,1][0])
  
lr_races.append(sum(lr_res)/len(lr_res))
rf_races.append(sum(rf_res)/len(rf_res))


lr_res = []
rf_res = []
for i in range(100):
        age = 65+np.random.randint(30)
        ELIX = [np.random.randint(2) for i in range(29)]
        G = [np.random.randint(2) for i in range(22)]
        drugs = [np.random.randint(30) for i in range(12)] 
        A = 0
        AmN = 0
        B = 0
        H = 0
        O = 0
        U = 1
        W = 0
        
        d = [age] + ELIX + G + drugs + [A, AmN, B, H, O, U, W]
        dat = [d]
        lr_res.append(lr.predict_proba(dat)[:,1][0])
        rf_res.append(rf.predict_proba(dat)[:,1][0])
  
lr_races.append(sum(lr_res)/len(lr_res))
rf_races.append(sum(rf_res)/len(rf_res))


lr_res = []
rf_res = []
for i in range(100):
        age = 65+np.random.randint(30)
        ELIX = [np.random.randint(2) for i in range(29)]
        G = [np.random.randint(2) for i in range(22)]
        drugs = [np.random.randint(30) for i in range(12)] 
        A = 0
        AmN = 0
        B = 0
        H = 0
        O = 0
        U = 0
        W = 1
        
        d = [age] + ELIX + G + drugs + [A, AmN, B, H, O, U, W]
        dat = [d]
        lr_res.append(lr.predict_proba(dat)[:,1][0])
        rf_res.append(rf.predict_proba(dat)[:,1][0])
  
lr_races.append(sum(lr_res)/len(lr_res))
rf_races.append(sum(rf_res)/len(rf_res))


# In[47]:


# Race influences Logistic model than random forest
plt.bar(range(0,len(lr_races)*2,2), lr_races, color='b')
plt.bar(range(1,len(lr_races)*2,2), rf_races, color='r')
plt.xticks(range(0,len(lr_races)*2,2), p1.index)
plt.show()


# #### Generate random matrix with 0.1 probability of changing a value

# In[48]:


trD = pd.DataFrame(tr)
sz=len(trD.index)


# In[49]:


ELIX = (np.random.rand(sz,29) < 0.1).astype('int')


# In[50]:


trD.loc[:][['ELIX1','ELIX2','ELIX3','ELIX4','ELIX5','ELIX6','ELIX7','ELIX8','ELIX9','ELIX10','ELIX11','ELIX12',
     'ELIX13','ELIX14','ELIX15','ELIX16','ELIX17','ELIX18','ELIX19','ELIX20','ELIX21','ELIX22','ELIX23',
     'ELIX24','ELIX25','ELIX26','ELIX27','ELIX28','ELIX29']] -= ELIX


trD.loc[:][['ELIX1','ELIX2','ELIX3','ELIX4','ELIX5','ELIX6','ELIX7','ELIX8','ELIX9','ELIX10','ELIX11','ELIX12',
     'ELIX13','ELIX14','ELIX15','ELIX16','ELIX17','ELIX18','ELIX19','ELIX20','ELIX21','ELIX22','ELIX23',
     'ELIX24','ELIX25','ELIX26','ELIX27','ELIX28','ELIX29']] = abs(trD.loc[:][['ELIX1','ELIX2','ELIX3','ELIX4','ELIX5','ELIX6','ELIX7','ELIX8','ELIX9','ELIX10','ELIX11','ELIX12',
     'ELIX13','ELIX14','ELIX15','ELIX16','ELIX17','ELIX18','ELIX19','ELIX20','ELIX21','ELIX22','ELIX23',
     'ELIX24','ELIX25','ELIX26','ELIX27','ELIX28','ELIX29']])


# In[51]:


G = (np.random.rand(sz,22) < 0.1).astype('int')

# modify the binary values
trD.loc[:][['G-2','G-3','G-4','G-5','G-6','G-7','G-8','G-9','G-10','G-11','G-12',
     'G-13','G-14','G-15','G-16','G-17','G-18','G-19','G-20','G-21','G-22','G-23']] -= G

# take absolute value to fix -1s
trD.loc[:][['G-2','G-3','G-4','G-5','G-6','G-7','G-8','G-9','G-10','G-11','G-12',
     'G-13','G-14','G-15','G-16','G-17','G-18','G-19','G-20','G-21','G-22','G-23']] = abs(trD.loc[:][['G-2','G-3','G-4','G-5','G-6','G-7','G-8','G-9','G-10','G-11','G-12',
     'G-13','G-14','G-15','G-16','G-17','G-18','G-19','G-20','G-21','G-22','G-23']])


# In[52]:


# Standard deviation of age
ageDv = trD['age'].std()
ageDv


# In[53]:


trD['age'].head()


# In[54]:


# Random distribution of age
trD['age']+np.random.normal(0,ageDv,trD['age'].size)


# In[55]:


d = ['drugs_m0-1', 'drugs_m1-2', 'drugs_m2-3', 'drugs_m3-4', 'drugs_m4-5', 'drugs_m5-6', 'drugs_m6-7', 
              'drugs_m7-8', 'drugs_m8-9', 'drugs_m9-10', 'drugs_m10-11','drugs_m11-12']
drugsDv = trD[d].std(axis=0)
drugsDv


# In[56]:


trD[d]=trD[d]+np.random.normal(0,1,(trD.index.size,12))*np.array(drugsDv)
trD[d].head()


# In[57]:


trD[d] = np.where(trD[d]<0,0,trD[d])
trD[d].head()


# In[581]:


def distort(fff):
    trD = pd.DataFrame(fff)
    sz=len(trD.index)
    ELIX = (np.random.rand(sz,29) < 0.05).astype('int')
    trD.loc[:][['ELIX1','ELIX2','ELIX3','ELIX4','ELIX5','ELIX6','ELIX7','ELIX8','ELIX9','ELIX10','ELIX11','ELIX12',
     'ELIX13','ELIX14','ELIX15','ELIX16','ELIX17','ELIX18','ELIX19','ELIX20','ELIX21','ELIX22','ELIX23',
     'ELIX24','ELIX25','ELIX26','ELIX27','ELIX28','ELIX29']] -= ELIX

    trD.loc[:][['ELIX1','ELIX2','ELIX3','ELIX4','ELIX5','ELIX6','ELIX7','ELIX8','ELIX9','ELIX10','ELIX11','ELIX12',
     'ELIX13','ELIX14','ELIX15','ELIX16','ELIX17','ELIX18','ELIX19','ELIX20','ELIX21','ELIX22','ELIX23',
     'ELIX24','ELIX25','ELIX26','ELIX27','ELIX28','ELIX29']] = abs(trD.loc[:][['ELIX1','ELIX2','ELIX3','ELIX4','ELIX5','ELIX6','ELIX7','ELIX8','ELIX9','ELIX10','ELIX11','ELIX12',
     'ELIX13','ELIX14','ELIX15','ELIX16','ELIX17','ELIX18','ELIX19','ELIX20','ELIX21','ELIX22','ELIX23',
     'ELIX24','ELIX25','ELIX26','ELIX27','ELIX28','ELIX29']])
    
    G = (np.random.rand(sz,22) < 0.05).astype('int')

    # modify the binary values
    trD.loc[:][['G-2','G-3','G-4','G-5','G-6','G-7','G-8','G-9','G-10','G-11','G-12',
     'G-13','G-14','G-15','G-16','G-17','G-18','G-19','G-20','G-21','G-22','G-23']] -= G

    # take absolute value to fix -1s
    trD.loc[:][['G-2','G-3','G-4','G-5','G-6','G-7','G-8','G-9','G-10','G-11','G-12',
     'G-13','G-14','G-15','G-16','G-17','G-18','G-19','G-20','G-21','G-22','G-23']] = abs(trD.loc[:][['G-2','G-3','G-4','G-5','G-6','G-7','G-8','G-9','G-10','G-11','G-12',
     'G-13','G-14','G-15','G-16','G-17','G-18','G-19','G-20','G-21','G-22','G-23']])

    ageDv = trD['age'].std()
    trD['age']+np.random.normal(0,ageDv,trD['age'].size)
    
    d = ['drugs_m0-1', 'drugs_m1-2', 'drugs_m2-3', 'drugs_m3-4', 'drugs_m4-5', 'drugs_m5-6', 'drugs_m6-7', 
              'drugs_m7-8', 'drugs_m8-9', 'drugs_m9-10', 'drugs_m10-11','drugs_m11-12']
    drugsDv = trD[d].std(axis=0)
    trD[d]=trD[d]+np.random.normal(0,1,(trD.index.size,12))*np.array(drugsDv)
    trD[d] = np.where(trD[d]<0,0,trD[d])
    return trD
    


# In[582]:


dst = []
for i in range(10): 
    dst.append( distort(tr))


# In[583]:


dstTr = pd.concat(dst)


# In[584]:


probs_l = lr.predict_proba(dstTr[cls])
probs_r = rf.predict_proba(dstTr[cls])


# In[585]:


tr_res = pd.DataFrame()
tr_res['age'] = dstTr['age']
tr_res['lr'] = probs_l[:,1]
tr_res['rf'] = probs_r[:,1]
gr=tr_res.groupby('age')

plt.plot(range(mina, maxa+1), list(gr['lr'].mean()))
plt.plot(range(mina, maxa+1), list(gr['rf'].mean()))


# ### 1. Test how logistic and random forest models behave outside the original 65-90 age range. It is reasonable to assume that for such patients the models will perform correctly?

# ### Ans - It would not be reasonable to assume that for patients outside the original age range 65-90 the model will perform correctly because the model was never trained on ages beyond the original 65-90 age range. The trend falls down showing following the same pattern as the 65-90 age range.

# In[127]:


lr_ages = []
rf_ages = []
for age in range(18,100):
    lr_res = []
    rf_res = []
    for i in range(1000):
        #age = 65
        ELIX = [np.random.randint(2) for i in range(29)]
        G = [np.random.randint(2) for i in range(22)]
        drugs = [np.random.randint(30) for i in range(12)] 
        A = 0
        AmN = 0
        B = 0
        H = 0
        O = 0
        U = 0
        W = 0
        r = np.random.randint(7)
        if r == 0:
            A = 1
        if r == 1:
            AmN = 1
        if r == 2:
            B = 1
        if r == 3:
            H = 1
        if r == 4:
            O = 1
        if r == 5:
            U = 1
        if r == 6:
            W = 1    
        d = [age] + ELIX + G + drugs + [A, AmN, B, H, O, U, W]
        dat = [d]
        lr_res.append(lr.predict_proba(dat)[:,1][0])
        rf_res.append(rf.predict_proba(dat)[:,1][0])
    lr_ages.append(sum(lr_res)/len(lr_res))
    rf_ages.append(sum(rf_res)/len(rf_res))


# In[128]:


plt.plot(range(18,100), lr_ages)
plt.plot(range(18,100), rf_ages)


# ### 2. Select randomly one patient from the data. Apply distortion of data to the patient 1000 times. Analyze probabilities output from the model for the distorted patient as compared with the original patient. Does the prediction/class change?

# #### Selecting a single random patient

# In[400]:


new = tr
rand_patient = pd.DataFrame()
rand_patient = new[new['index'] == np.random.choice(new.index)]
rand_patient


# In[403]:


rand_pat =pd.DataFrame
rand_pat = rand_patient.append([rand_patient]*1000,ignore_index=True)
rand_pat.count().head()


# #### Probabilities of ELIX codes

# In[404]:


INDEX1 = pd.DataFrame(tr.groupby('ELIX1')['index'].size().div(len(tr)).reset_index(name='ELIX_1'))
INDEX2 = pd.DataFrame(tr.groupby('ELIX2')['index'].size().div(len(tr)).reset_index(name='ELIX_2'))
INDEX3 = pd.DataFrame(tr.groupby('ELIX3')['index'].size().div(len(tr)).reset_index(name='ELIX_3'))
INDEX4 = pd.DataFrame(tr.groupby('ELIX4')['index'].size().div(len(tr)).reset_index(name='ELIX_4'))
INDEX5 = pd.DataFrame(tr.groupby('ELIX5')['index'].size().div(len(tr)).reset_index(name='ELIX_5'))
INDEX6 = pd.DataFrame(tr.groupby('ELIX6')['index'].size().div(len(tr)).reset_index(name='ELIX_6'))
INDEX7 = pd.DataFrame(tr.groupby('ELIX7')['index'].size().div(len(tr)).reset_index(name='ELIX_7'))
INDEX8 = pd.DataFrame(tr.groupby('ELIX8')['index'].size().div(len(tr)).reset_index(name='ELIX_8'))
INDEX9 = pd.DataFrame(tr.groupby('ELIX9')['index'].size().div(len(tr)).reset_index(name='ELIX_9'))
INDEX10 = pd.DataFrame(tr.groupby('ELIX10')['index'].size().div(len(tr)).reset_index(name='ELIX_10'))
INDEX11 = pd.DataFrame(tr.groupby('ELIX11')['index'].size().div(len(tr)).reset_index(name='ELIX_11'))
INDEX12 = pd.DataFrame(tr.groupby('ELIX12')['index'].size().div(len(tr)).reset_index(name='ELIX_12'))
INDEX13 = pd.DataFrame(tr.groupby('ELIX13')['index'].size().div(len(tr)).reset_index(name='ELIX_13'))
INDEX14 = pd.DataFrame(tr.groupby('ELIX14')['index'].size().div(len(tr)).reset_index(name='ELIX_14'))
INDEX15 = pd.DataFrame(tr.groupby('ELIX15')['index'].size().div(len(tr)).reset_index(name='ELIX_15'))
INDEX16 = pd.DataFrame(tr.groupby('ELIX16')['index'].size().div(len(tr)).reset_index(name='ELIX_16'))
INDEX17 = pd.DataFrame(tr.groupby('ELIX17')['index'].size().div(len(tr)).reset_index(name='ELIX_17'))
INDEX18 = pd.DataFrame(tr.groupby('ELIX18')['index'].size().div(len(tr)).reset_index(name='ELIX_18'))
INDEX19 = pd.DataFrame(tr.groupby('ELIX19')['index'].size().div(len(tr)).reset_index(name='ELIX_19'))
INDEX20 = pd.DataFrame(tr.groupby('ELIX20')['index'].size().div(len(tr)).reset_index(name='ELIX_20'))
INDEX21 = pd.DataFrame(tr.groupby('ELIX21')['index'].size().div(len(tr)).reset_index(name='ELIX_21'))
INDEX22 = pd.DataFrame(tr.groupby('ELIX22')['index'].size().div(len(tr)).reset_index(name='ELIX_22'))
INDEX23 = pd.DataFrame(tr.groupby('ELIX23')['index'].size().div(len(tr)).reset_index(name='ELIX_23'))
INDEX24 = pd.DataFrame(tr.groupby('ELIX24')['index'].size().div(len(tr)).reset_index(name='ELIX_24'))
INDEX25 = pd.DataFrame(tr.groupby('ELIX25')['index'].size().div(len(tr)).reset_index(name='ELIX_25'))
INDEX26 = pd.DataFrame(tr.groupby('ELIX26')['index'].size().div(len(tr)).reset_index(name='ELIX_26'))
INDEX27 = pd.DataFrame(tr.groupby('ELIX27')['index'].size().div(len(tr)).reset_index(name='ELIX_27'))
INDEX28 = pd.DataFrame(tr.groupby('ELIX28')['index'].size().div(len(tr)).reset_index(name='ELIX_28'))
INDEX29 = pd.DataFrame(tr.groupby('ELIX29')['index'].size().div(len(tr)).reset_index(name='ELIX_29'))

INDEX = pd.concat([INDEX1,INDEX2,INDEX3,INDEX4,INDEX5,INDEX6,INDEX7,INDEX8,INDEX9,INDEX10,INDEX11,INDEX12,INDEX13,INDEX14,
                   INDEX15,INDEX16,INDEX17,INDEX18,INDEX19,INDEX20,INDEX21,INDEX22,INDEX23,INDEX24,INDEX25,INDEX26,INDEX27,
                   INDEX28,INDEX29],
                  axis=1,sort=False)
INDEX


# #### Probabilities for procedural codes

# In[405]:


G2 = pd.DataFrame(tr.groupby('G-2')['index'].size().div(len(tr)).reset_index(name='G_2'))
G3 = pd.DataFrame(tr.groupby('G-3')['index'].size().div(len(tr)).reset_index(name='G_3'))
G4 = pd.DataFrame(tr.groupby('G-4')['index'].size().div(len(tr)).reset_index(name='G_4'))
G5 = pd.DataFrame(tr.groupby('G-5')['index'].size().div(len(tr)).reset_index(name='G_5'))
G6 = pd.DataFrame(tr.groupby('G-6')['index'].size().div(len(tr)).reset_index(name='G_6'))
G7 = pd.DataFrame(tr.groupby('G-7')['index'].size().div(len(tr)).reset_index(name='G_7'))
G8 = pd.DataFrame(tr.groupby('G-8')['index'].size().div(len(tr)).reset_index(name='G_8'))
G9 = pd.DataFrame(tr.groupby('G-9')['index'].size().div(len(tr)).reset_index(name='G_9'))
G10 = pd.DataFrame(tr.groupby('G-10')['index'].size().div(len(tr)).reset_index(name='G_10'))
G11 = pd.DataFrame(tr.groupby('G-11')['index'].size().div(len(tr)).reset_index(name='G_11'))
G12 = pd.DataFrame(tr.groupby('G-12')['index'].size().div(len(tr)).reset_index(name='G_12'))
G13 = pd.DataFrame(tr.groupby('G-13')['index'].size().div(len(tr)).reset_index(name='G_13'))
G14 = pd.DataFrame(tr.groupby('G-14')['index'].size().div(len(tr)).reset_index(name='G_14'))
G15 = pd.DataFrame(tr.groupby('G-15')['index'].size().div(len(tr)).reset_index(name='G_15'))
G16 = pd.DataFrame(tr.groupby('G-16')['index'].size().div(len(tr)).reset_index(name='G_16'))
G17 = pd.DataFrame(tr.groupby('G-17')['index'].size().div(len(tr)).reset_index(name='G_17'))
G18 = pd.DataFrame(tr.groupby('G-18')['index'].size().div(len(tr)).reset_index(name='G_18'))
G19 = pd.DataFrame(tr.groupby('G-19')['index'].size().div(len(tr)).reset_index(name='G_19'))
G20 = pd.DataFrame(tr.groupby('G-20')['index'].size().div(len(tr)).reset_index(name='G_20'))
G21 = pd.DataFrame(tr.groupby('G-21')['index'].size().div(len(tr)).reset_index(name='G_21'))
G22 = pd.DataFrame(tr.groupby('G-22')['index'].size().div(len(tr)).reset_index(name='G_22'))
G23 = pd.DataFrame(tr.groupby('G-23')['index'].size().div(len(tr)).reset_index(name='G_23'))

G = pd.concat([G2,G3,G4,G5,G6,G7,G8,G9,G10,G11,G12,G13,G14,
                   G15,G16,G17,G18,G19,G20,G21,G22,G23],
                  axis=1,sort=False)
G


# In[574]:


def distort2(fff):
    trD = pd.DataFrame(fff)
    sz = len(trD.index)
    ELIX1 = (np.random.rand(sz,1) < 0.17).astype('int')
    trD.loc[:][['ELIX1']] -= ELIX1
    ELIX2 = (np.random.rand(sz,1) < 0.002).astype('int')
    trD.loc[:][['ELIX2']] -= ELIX2
    ELIX3 = (np.random.rand(sz,1) < 0.065).astype('int')
    trD.loc[:][['ELIX3']] -= ELIX3
    ELIX4 = (np.random.rand(sz,1) < 0.079).astype('int')
    trD.loc[:][['ELIX4']] -= ELIX4
    ELIX5 = (np.random.rand(sz,1) < 0.078).astype('int')
    trD.loc[:][['ELIX5']] -= ELIX5
    ELIX6 = (np.random.rand(sz,1) < 0.172).astype('int')
    trD.loc[:][['ELIX6']] -= ELIX6
    ELIX7 = (np.random.rand(sz,1) < 0.059).astype('int')
    trD.loc[:][['ELIX7']] -= ELIX7
    ELIX8 = (np.random.rand(sz,1) < 0.160).astype('int')
    trD.loc[:][['ELIX8']] -= ELIX8
    ELIX9 = (np.random.rand(sz,1) < 0.379).astype('int')
    trD.loc[:][['ELIX9']] -= ELIX9
    ELIX10 = (np.random.rand(sz,1) < 0.006).astype('int')
    trD.loc[:][['ELIX10']] -= ELIX10
    ELIX11 = (np.random.rand(sz,1) < 0.061).astype('int')
    trD.loc[:][['ELIX11']] -= ELIX11
    ELIX12 = (np.random.rand(sz,1) < 0.160).astype('int')
    trD.loc[:][['ELIX12']] -= ELIX12
    ELIX13 = (np.random.rand(sz,1) < 0.164).astype('int')
    trD.loc[:][['ELIX13']] -= ELIX13
    ELIX14 = (np.random.rand(sz,1) < 0.265).astype('int')
    trD.loc[:][['ELIX14']] -= ELIX14
    ELIX15 = (np.random.rand(sz,1) < 0.115).astype('int')
    trD.loc[:][['ELIX15']] -= ELIX15
    ELIX16 = (np.random.rand(sz,1) < 0.095).astype('int')
    trD.loc[:][['ELIX16']] -= ELIX16
    ELIX17 = (np.random.rand(sz,1) < 0.006).astype('int')
    trD.loc[:][['ELIX17']] -= ELIX17
    ELIX18 = (np.random.rand(sz,1) < 0.190).astype('int')
    trD.loc[:][['ELIX18']] -= ELIX18
    ELIX19 = (np.random.rand(sz,1) < 0.032).astype('int')
    trD.loc[:][['ELIX19']] -= ELIX19
    ELIX20 = (np.random.rand(sz,1) < 0.03).astype('int')
    trD.loc[:][['ELIX20']] -= ELIX20
    ELIX21 = (np.random.rand(sz,1) < 0.135).astype('int')
    trD.loc[:][['ELIX21']] -= ELIX21
    ELIX22 = (np.random.rand(sz,1) < 0.007).astype('int')
    trD.loc[:][['ELIX22']] -= ELIX22
    ELIX23 = (np.random.rand(sz,1) < 0.002).astype('int')
    trD.loc[:][['ELIX23']] -= ELIX23
    ELIX24 = (np.random.rand(sz,1) < 0.07).astype('int')
    trD.loc[:][['ELIX24']] -= ELIX24
    ELIX25 = (np.random.rand(sz,1) < 0.019).astype('int')
    trD.loc[:][['ELIX25']] -= ELIX25
    ELIX26 = (np.random.rand(sz,1) < 0.792).astype('int')
    trD.loc[:][['ELIX26']] -= ELIX26
    ELIX27 = (np.random.rand(sz,1) < 0.217).astype('int')
    trD.loc[:][['ELIX27']] -= ELIX27
    ELIX28 = (np.random.rand(sz,1) < 0.028).astype('int')
    trD.loc[:][['ELIX28']] -= ELIX28
    ELIX29 = (np.random.rand(sz,1) < 0.241).astype('int')
    trD.loc[:][['ELIX29']] -= ELIX29

    trD.loc[:][['ELIX1','ELIX2','ELIX3','ELIX4','ELIX5','ELIX6','ELIX7','ELIX8','ELIX9','ELIX10','ELIX11','ELIX12',
         'ELIX13','ELIX14','ELIX15','ELIX16','ELIX17','ELIX18','ELIX19','ELIX20','ELIX21','ELIX22','ELIX23',
         'ELIX24','ELIX25','ELIX26','ELIX27','ELIX28','ELIX29']] = abs(trD.loc[:][['ELIX1','ELIX2','ELIX3','ELIX4',
         'ELIX5','ELIX6','ELIX7','ELIX8','ELIX9','ELIX10','ELIX11','ELIX12',
         'ELIX13','ELIX14','ELIX15','ELIX16','ELIX17','ELIX18','ELIX19','ELIX20','ELIX21','ELIX22','ELIX23',
         'ELIX24','ELIX25','ELIX26','ELIX27','ELIX28','ELIX29']])

    G2 = (np.random.rand(sz,1) < 0.127).astype('int')
    trD.loc[:][['G-2']] -= G2
    G3 = (np.random.rand(sz,1) < 0.049).astype('int')
    trD.loc[:][['G-3']] -= G3
    G4 = (np.random.rand(sz,1) < 0.000022).astype('int')
    trD.loc[:][['G-4']] -= G4
    G5 = (np.random.rand(sz,1) < 0.003).astype('int')
    trD.loc[:][['G-5']] -= G5
    G6 = (np.random.rand(sz,1) < 0.767).astype('int')
    trD.loc[:][['G-6']] -= G6
    G7 = (np.random.rand(sz,1) < 0.047).astype('int')
    trD.loc[:][['G-7']] -= G7
    G8 = (np.random.rand(sz,1) < 0.00069).astype('int')
    trD.loc[:][['G-8']] -= G8
    G9 = (np.random.rand(sz,1) < 0.0019).astype('int')
    trD.loc[:][['G-9']] -= G9
    G10 = (np.random.rand(sz,1) < 0.000077).astype('int')
    trD.loc[:][['G-10']] -= G10
    G11 = (np.random.rand(sz,1) < 0.598).astype('int')
    trD.loc[:][['G-11']] -= G11
    G12 = (np.random.rand(sz,1) < 0.153).astype('int')
    trD.loc[:][['G-12']] -= G12
    G13 = (np.random.rand(sz,1) < 0.872).astype('int')
    trD.loc[:][['G-13']] -= G13
    G14 = (np.random.rand(sz,1) < 0.000133).astype('int')
    trD.loc[:][['G-14']] -= G14
    G15 = (np.random.rand(sz,1) < 0.000022).astype('int')
    trD.loc[:][['G-15']] -= G15
    G16 = (np.random.rand(sz,1) < 0.0013).astype('int')
    trD.loc[:][['G-16']] -= G16
    G17 = (np.random.rand(sz,1) < 0.3325).astype('int')
    trD.loc[:][['G-17']] -= G17
    G18 = (np.random.rand(sz,1) < 0.719).astype('int')
    trD.loc[:][['G-18']] -= G18
    G19 = (np.random.rand(sz,1) < 0.000011).astype('int')
    trD.loc[:][['G-19']] -= G19
    G20 = (np.random.rand(sz,1) < 0.000431).astype('int')
    trD.loc[:][['G-20']] -= G20
    G21 = (np.random.rand(sz,1) < 0.472).astype('int')
    trD.loc[:][['G-21']] -= G21
    G22 = (np.random.rand(sz,1) < 0.211).astype('int')
    trD.loc[:][['G-22']] -= G22
    G23 = (np.random.rand(sz,1) < 0.000022).astype('int')
    trD.loc[:][['G-23']] -= G23
           
    # take absolute value to fix -1s
    trD.loc[:][['G-2','G-3','G-4','G-5','G-6','G-7','G-8','G-9','G-10','G-11','G-12',
                'G-13','G-14','G-15','G-16','G-17','G-18','G-19','G-20','G-21','G-22','G-23']] = abs(trD.loc[:][['G-2','G-3',
                'G-4','G-5','G-6','G-7','G-8','G-9','G-10','G-11','G-12',
                'G-13','G-14','G-15','G-16','G-17','G-18','G-19','G-20','G-21','G-22','G-23']])
    
    ageDv = tr['age'].std()
    trD['age']=round(trD['age']+np.random.normal(0,ageDv,trD['age'].size))

    d = ['drugs_m0-1', 'drugs_m1-2', 'drugs_m2-3', 'drugs_m3-4', 'drugs_m4-5', 'drugs_m5-6', 'drugs_m6-7', 
         'drugs_m7-8', 'drugs_m8-9', 'drugs_m9-10', 'drugs_m10-11','drugs_m11-12']
    drugsDv = tr[d].std(axis=0)
    trD[d]=trD[d]+np.random.normal(0,1,(trD.index.size,12))*np.array(drugsDv)
    trD[d] = np.where(trD[d]<0,0,round(trD[d]))
    return trD


# In[575]:


dst = []
dst.append(distort2(rand_pat))


# In[576]:


dstTr = pd.concat(dst)
dstTr.head()


# In[577]:


probs_l = lr.predict_proba(dstTr[cls])
probs_r = rf.predict_proba(dstTr[cls])


# In[580]:


tr_res = pd.DataFrame()
tr_res['age'] = dstTr['age']
tr_res['lr'] = probs_l[:,1]
tr_res['rf'] = probs_r[:,1]
gr=tr_res.groupby('age')

plt.plot(list(gr['lr'].mean()),color='b')
plt.plot(list(gr['rf'].mean()),color='r')
plt.xlim(65,90)


# In[ ]:




