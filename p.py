# https://zhuanlan.zhihu.com/p/99579173
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


import seaborn as sns



plt.rcParams['font.sans-serif']=['SimHei'] #用於在數據集中正常顯示中文字符
plt.rcParams['axes.unicode_minus']=False 

#read dataset
data = pd.read_csv(r"C:\Users\09333\Desktop\py\dataminning\com(all).csv")


# data.info()


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

# data.info()


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



# 独热编码（One-Hot Encoding），又称一位有效编码，其方法是使用N位状态寄存器来对N个状态进行编码，
# 每个状态都有它独立的寄存器位，并且在任意时候，其中只有一位有效。即，只有一位是1，其余都是零值。


# DataFrame 是一个表格型的数据结构

# category

# reshape(-1, 1)：变成只有一列

#sklearn里的封装好的各种算法使用前都要fit，fit相对于整个代码而言，为后续API服务。fit之后，然后调用各种API方法


scaleronehot=preprocessing.OneHotEncoder().fit(np.array(data.cat.value_counts().index).reshape(-1, 1))
# return fitted obj
# print(data.cat.value_counts().index)
# Index(['电子商务', '企业服务', '本地生活', '金融', '社交网络', '文娱传媒', '教育', '工具软件', '游戏',
#        '汽车交通', '旅游', '医疗健康', '硬件', '广告营销', '房产服务', '体育运动', '新工业', '物流', '农业'],  19种
#       dtype='object')

data2=pd.DataFrame(scaleronehot.transform(np.array(data.cat).reshape(-1, 1)).toarray(),columns=range(0,19))
# Transform X using one-hot encoding.
#  data2:  [6272 rows x 19 columns]


# financing
scaleronehot2=preprocessing.OneHotEncoder().fit(np.array(data.financing.value_counts().index).reshape(-1, 1))
data3=pd.DataFrame(scaleronehot2.transform(np.array(data.financing).reshape(-1, 1)).toarray(),columns=range(19,35))



# comany_address
scaleronehot3=preprocessing.OneHotEncoder().fit(np.array(data.com_addr.value_counts().index).reshape(-1, 1))
data4=pd.DataFrame(scaleronehot3.transform(np.array(data.com_addr).reshape(-1, 1)).toarray(),columns=range(35,72))



# 用来连接DataFrame对象
# axis=1  ，横向表拼接   6272 rows 
data5=pd.concat([data2,data3,data4,data.live_days],axis=1) 
# [6272 rows x 73 columns]
# print(data5)


# corr() 函数用于返回 DataFrame 的相关系数矩阵（Correlation matrix
corr=data5.corr()
plt.figure(figsize=(10,10))
# figsize:指定figure的宽和高(英寸)
sns.heatmap(corr,cmap="BuGn")

# plt.ylim(73,0)
# y轴范围

#corr :[73 rows x 73 columns]
# plt.show()


# 热力图越深色为越强相关


corr.live_days.sort_values()[-6:]

# 20           0.054765
# 39           0.055830
# 23           0.056662
# 22           0.077564
# 31           0.094593
# live_days    1.000000

# 区间
# 0~18 领域   
# 19~34 融资方法
# 35~71 公司地点
# 72 存活天数



print(data.financing.value_counts().index[31-19])
print(data.financing.value_counts().index[22-19])
print(data.financing.value_counts().index[23-19])

print(data.com_addr.value_counts().index[39-35])
print(data.financing.value_counts().index[20-19])




# print(corr)
# corr[num] 按列（column）为key查询
level=0.1
for row in range(0,72):
    for column in range(0,72):
        if ((corr.loc[row,column]>=level) & (corr.loc[row,column]<1)):
            
            if(row<=18):
                if(column<=18):
                    print(data.cat.value_counts().index[row]," : ",data.cat.value_counts().index[column]," = ",corr.loc[row,column])
                elif(column<=34):
                    print(data.cat.value_counts().index[row]," : ",data.financing.value_counts().index[column-19]," = ",corr.loc[row,column])
                elif(column>=35):
                    print(data.cat.value_counts().index[row]," : ",data.com_addr.value_counts().index[column-35]," = ",corr.loc[row,column])
            elif(row<=34):
                if(column<=18):
                    print(data.financing.value_counts().index[row-19]," : ",data.cat.value_counts().index[column]," = ",corr.loc[row,column])
                elif(column<=34):
                    print(data.financing.value_counts().index[row-19]," : ",data.financing.value_counts().index[column-19]," = ",corr.loc[row,column])
                elif(column>=35):
                    print(data.financing.value_counts().index[row-19]," : ",data.com_addr.value_counts().index[column-35]," = ",corr.loc[row,column])
            elif(row>=35):
                if(column<=18):
                    print(data.com_addr.value_counts().index[row-35]," : ",data.cat.value_counts().index[column]," = ",corr.loc[row,column])
                elif(column<=34):
                    print(data.com_addr.value_counts().index[row-35]," : ",data.financing.value_counts().index[column-19]," = ",corr.loc[row,column])
                elif(column>=35):
                    print(data.com_addr.value_counts().index[row-35]," : ",data.com_addr.value_counts().index[column-35]," = ",corr.loc[row,column])
                    
# 弱相关：
# 企业服务  :  香港  =  0.1335995260407172
# A轮  :  福建  =  0.12507250066324943
# 种子轮  :  福建  =  0.14665307301514435
# B+轮  :  福建  =  0.16248528147042204