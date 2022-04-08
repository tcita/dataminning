# https://zhuanlan.zhihu.com/p/99579173
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


import seaborn as sns



plt.rcParams['font.sans-serif']=['SimHei'] #用於在數據集中正常顯示中文字符
plt.rcParams['axes.unicode_minus']=False 

#read dataset
data = pd.read_csv(r"C:\Users\09333\Desktop\py\dataminning\com(all).csv")


data.info()
#data cleaning
data = data.dropna(axis=1,how='all')
data = data.drop('bianh',axis=1)

data['se_cat'].fillna('unknow',inplace=True)
data['com_des'].fillna('unknow',inplace=True)
data['total_money'].fillna('0',inplace=True)
data['death_reason'].fillna('unknow',inplace=True)
data['invest_name'].fillna('unknow',inplace=True)
data['ceo_name'].fillna('unknow',inplace=True)
data['ceo_des'].fillna('unknow',inplace=True)
data['ceo_per_des'].fillna('unknow',inplace=True)
# data.isnull().sum()
data.info()


# 頻率直方圖
# data.sort_values(by ='live_days',ascending=False).head()
plt.figure(figsize=(10,6))
plt.hist(data.live_days,bins=100,color='c')
plt.xlim(0, 8000)
plt.xlabel('存活天數/天')
plt.ylabel('公司數量/個')



# plt.show()
# 第3,7年大量公司倒閉

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

plt.xlabel('公司死亡原因')
plt.ylabel('數量/個')



plt.figure(figsize=(10,6))

data.cat.value_counts().plot.pie(autopct="%1.1f%%")
# plt.legend()



# 热力图
# 分析存活时长与其他特征的关系（省份、领域、融资阶段）
# 热力图在实际中常用于展示一组变量的相关系数矩阵，
# 在展示列联表的数据分布上也有较大的用途，通过热力图我们可以非常直观地感受到数值大小的差异状况。

from sklearn  import preprocessing

scaleronehot=preprocessing.OneHotEncoder().fit(np.array(data.cat.value_counts().index).reshape(-1, 1))
data2=pd.DataFrame(scaleronehot.transform(np.array(data.cat).reshape(-1, 1)).toarray(),columns=range(0,19))

scaleronehot2=preprocessing.OneHotEncoder().fit(np.array(data.financing.value_counts().index).reshape(-1, 1))
data3=pd.DataFrame(scaleronehot2.transform(np.array(data.financing).reshape(-1, 1)).toarray(),columns=range(19,35))

scaleronehot3=preprocessing.OneHotEncoder().fit(np.array(data.com_addr.value_counts().index).reshape(-1, 1))
data4=pd.DataFrame(scaleronehot3.transform(np.array(data.com_addr).reshape(-1, 1)).toarray(),columns=range(35,72))

data5=pd.concat([data2,data3,data4,data.live_days],axis=1) 


corr=data5.corr()
plt.figure(figsize=(10,10))
sns.heatmap(corr)
plt.ylim(73,0)
plt.show()

corr.live_days.sort_values()[-6:]
print(data.financing.value_counts().index[31-19])
print(data.financing.value_counts().index[22-19])
print(data.financing.value_counts().index[23-19])
print(data.com_addr.value_counts().index[39-35])
print(data.financing.value_counts().index[20-19])