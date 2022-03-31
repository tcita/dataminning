# https://zhuanlan.zhihu.com/p/99579173
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签 
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号

data = pd.read_csv(r"C:\Users\09333\Desktop\py\com(all).csv")
data.info()

data = data.dropna(axis=1,how='all')
data = data.drop('bianh',axis=1)
data.isnull().sum()

data['se_cat'].fillna('unknow',inplace=True)
data['com_des'].fillna('unknow',inplace=True)
data['total_money'].fillna('0',inplace=True)
data['death_reason'].fillna('unknow',inplace=True)
data['invest_name'].fillna('unknow',inplace=True)
data['ceo_name'].fillna('unknow',inplace=True)
data['ceo_des'].fillna('unknow',inplace=True)
data['ceo_per_des'].fillna('unknow',inplace=True)

data.sort_values(by ='live_days',ascending=False).head()

# plt.figure(figsize=(10,6))
# plt.hist(data.live_days,bins=100,color='c')
# plt.xlim(0, 8000)
# plt.xlabel('survive days')
# plt.ylabel('the amount of companies')
# plt.show()

death_rs =[]
for death_r in data['death_reason']:
    if death_r != 'unknow':
        death_rs.extend(death_r.split(' '))
    else:
        continue
death_rs = pd.Series(data=death_rs)

death_x = death_rs.value_counts().index
death_y = death_rs.value_counts().values
plt.figure(figsize=(10,6))
plt.xticks(rotation=-60)
plt.bar(death_x[0:15],death_y[0:15],align='center',color='c')

plt.xlabel('the reason of company die')
plt.ylabel('amount')
plt.show()