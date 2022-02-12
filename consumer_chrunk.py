import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
consumer_Churn = pd.read_csv("C:/Users/Administrator/Downloads/Telco-Customer-Churn.csv")
pd.set_option('display.max_columns', None)
# print(consumer_Churn.info())    # (7043, 21)
# 用户id 性别 老年用户 伴侣用户 亲属用户 在网时长 电话服务 多线服务 互联网服务
# 网络安全服务 在线备份服务 设备保护服务 技术支持服务 网络电视 网络电影 签订合同方式
# 开通电子账单 付款方式 月费用 总费用 流失
# print(consumer_Churn.describe())
# print(consumer_chrunk.isnull().any())
print(consumer_Churn.shape)
# 总费用应该是float型，确实object型
consumer_Churn['TotalCharges'] = consumer_Churn['TotalCharges'].replace(' ', 0)  # 11
# 说明其有空字符串,查看其空字符串，都是未流失客户，所以用0填充
consumer_Churn['TotalCharges'] = consumer_Churn['TotalCharges'].astype('float64')
consumer_Churn['Churn'] = consumer_Churn['Churn'].replace({'No': 0, 'Yes': 1})
Churn_True = consumer_Churn[consumer_Churn['Churn'] == 1]
# print(consumer_Churn.describe())
# customerID	gender	SeniorCitizen	Partner	Dependents	tenure
# PhoneService	MultipleLines	InternetService	OnlineSecurity
# OnlineBackup	DeviceProtection	TechSupport	StreamingTV
# StreamingMovies	Contract	PaperlessBilling	PaymentMethod
# MonthlyCharges	TotalCharges	Churn
# 探索性别与年龄与客户流失率的关系
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
fig = plt.figure(figsize=(20, 10))
plt.tight_layout()
# print(consumer_Churn.head())

# 计算整体流失率，整体流失率为0.2653698707936959
Churn_p = Churn_True.shape[0] / consumer_Churn.shape[0]

# 计算不同变量下流失率的变化
def Churn_dif(vari):
    result = {}
    for i in consumer_Churn[vari].unique():
        result[i] = consumer_Churn[(consumer_Churn[vari] == i) & (consumer_Churn['Churn'] == 1)].shape[0] / \
                    consumer_Churn[consumer_Churn[vari] == i].shape[0]
    return result

# 从用户角度分析流失率
# for i, j in enumerate(['gender','SeniorCitizen','Partner','Dependents']):
#     ax1 = fig.add_subplot(2, 2, i+1)
#     ax = sns.countplot(x='Churn', hue=j, data=consumer_Churn, )
#     plot_data1 = consumer_Churn[j].value_counts()
#     plot_data2 = Churn_True[j].value_counts()
#     # print(plot_data1, plot_data2)
#     j_churn = Churn_dif(j)
#     j_factor = consumer_Churn[j].unique()
#     if len(j_factor) == 2:
#         start, span = 0.7, 0.4
#     elif len(j_factor) == 3:
#         start, span = 0.7, 0.25
#     else:
#         start, span = 0, 0
#     for m, n in enumerate(j_factor):
#         ax.annotate(s='%s' % round(j_churn[n], 2), xy=(start+m*span, 50), fontsize=14)
#     ax1.set_title(j)
# plt.show()

# 由图可得出结论：流失率与性别无关、老年用户的流失率偏高，有伴侣用户的流失率较低，
# 无伴侣用户流失率偏高，无亲属用户的流失率偏高，有亲属用户的流失率偏低

# 在网时长与用户流失率的关系
#
# consumer_Churn['tenure_bin'] = pd.cut(consumer_Churn['tenure'],36,labels=[i*2 for i in range(1,37)])
# tenure_churn = consumer_Churn[['tenure', 'tenure_bin', 'Churn']]
# # 计算以2为间隔的在网时长的客户流失率
# tenure_churn_per = tenure_churn.groupby(['tenure_bin'])['Churn'].sum() / tenure_churn.groupby(['tenure_bin'])['tenure'].count()
# ax2 = sns.lineplot(data=tenure_churn_per)
# ax2.axhline(y=Churn_p, ls=":", c='green')
# ax2.annotate(s='整体流失率:%s' % round(Churn_p,2), xy=(0.9,0.23), c='green')
# plt.show()
# print(consumer_Churn['tenure'].describe())
# min:0 max:72

# 有图可以看出在网时间越长，用户流失率越低，其中在网时间2个月以内的流失率最高

# 从产品角度分析流失率
# for i, j in enumerate(['PhoneService', 'MultipleLines',	'InternetService', 'OnlineSecurity',
#                        'OnlineBackup', 'DeviceProtection',	'TechSupport', 'StreamingTV', 'StreamingMovies']):
#     ax1 = fig.add_subplot(3, 3, i+1)
#     fig.subplots_adjust(hspace=0.8, wspace=0.3)
#     ax = sns.countplot(x=j, hue='Churn', data=consumer_Churn, )
#     j_churn = Churn_dif(j)
#     j_factor = consumer_Churn[j].unique()
#     if len(j_factor) == 2:
#         start, span = 0.1, 1
#     elif len(j_factor) == 3:
#         start, span = 0.02, 1
#     else:
#         start, span = 0, 0
#     for m, n in enumerate(j_factor):
#         ax.annotate(s='%s' % round(j_churn[n], 2), xy=(start+m*span, 50), fontsize=12)
#     ax1.set_title(j)
# plt.show()

# 由图可得出结论：电话服务和用户流失率无关，多线服务和用户流失率较高，互联网服务和用户流失率有关，
# 其中订购光纤服务的用户流失率明显偏高，在订购互联网服务中：其中不订购网络安全业务的用户流失率明显偏高
# 不订购在线备份服务的用户流失率偏高，不订购设备保护服务的用户流失率偏高，不订购技术支持服务的用户流失率偏高，
# 订不订购流媒体服务的用户流失率都偏高。


# 查看流失月费用和总费用 MonthlyCharges	TotalCharges
# ax3 = fig.add_subplot(2, 2, 1)
# ax3.hist(Churn_True.MonthlyCharges, bins=30)
# ax3.set_title('流失用户月消费')
# ax4 = fig.add_subplot(2, 2, 2)
# ax4.hist(Churn_True.TotalCharges, bins=30)
# ax4.set_title('流失用户总消费')
# plt.show()

# 由图可知流失用户月费用主要集中在70~110之间

# 从订购方式来分析用户流失率
# for i, j in enumerate(['Contract', 'PaperlessBilling', 'PaymentMethod']):
#     ax1 = fig.add_subplot(2, 2, i+1)
#     fig.subplots_adjust(hspace=0.8, wspace=0.3)
#     ax = sns.countplot(x=j, hue='Churn', data=consumer_Churn, )
#     j_churn = Churn_dif(j)
#     j_factor = consumer_Churn[j].unique()
#     if len(j_factor) == 2:
#         start, span = 0.1, 1
#     elif len(j_factor) == 3:
#         start, span = 0.02, 1
#     elif len(j_factor) == 4:
#         start, span = 0.02, 1
#     for m, n in enumerate(j_factor):
#         ax.annotate(s='%s' % round(j_churn[n], 2), xy=(start+m*span, 50), fontsize=12)
#     ax1.set_title(j)
#     ax1.tick_params(axis="y", labelsize=10)
#     ax1.tick_params(axis="x", labelsize=8)
# plt.show()

# 由图表可以看出按月订购服务的用户流失率偏高，开通电子账单的用户流失率偏高，使用电子支票方式支付的用户流失率最高

# 从高流失用户群体中挖掘出未流失用户的消费模型，在网时长大于24个小时，月消费大于60的高质量用户
# 老年用户的流失率偏高，
# 无伴侣用户流失率偏高，无亲属用户的流失率偏高，多线服务和用户流失率较高，互联网服务和用户流失率有关，
# 按月订购服务的用户流失率偏高，开通电子账单的用户流失率偏高，使用电子支票方式支付的用户流失率最高
# 用户是无法改变的所以探索其消费模型

# gender	SeniorCitizen	Partner	Dependents	tenure
# PhoneService	MultipleLines	InternetService	OnlineSecurity
# OnlineBackup	DeviceProtection	TechSupport	StreamingTV
# StreamingMovies	Contract	PaperlessBilling	PaymentMethod
# MonthlyCharges	TotalCharges	Churn
def high_quality_user(var, value):
    high_quality = consumer_Churn[(consumer_Churn['tenure'] >= 24) & (consumer_Churn['MonthlyCharges'] >= 60)
                & (consumer_Churn['Churn'] == 0) & (consumer_Churn[var] == value)]
    result_dic = {}
    for i in ['PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup','DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']:
        if i == 'InternetService':
            for j in high_quality[i].unique():
                result_dic[j] = high_quality[high_quality['InternetService'] == j].shape[0] / high_quality.shape[0]
        else:
            result_dic[i] = high_quality[high_quality[i] == 'Yes'].shape[0] / high_quality.shape[0]
    return result_dic

import numpy as np
def high_quality_service(var, value):
    result = high_quality_user(var, value)
    labels = np.array(list(result.keys()))
    stats = np.array(list(result.values()))
    angles = np.linspace(0, 2*np.pi, labels.size, endpoint=False)
    stats = np.concatenate((stats, [stats[0]]))
    angles = np.concatenate((angles, [angles[0]]))
    ax = fig.add_subplot(111, polar=True)
    ax.plot(angles, stats, 'o-', linewidth=2)
    ax.fill(angles, stats, alpha=0.25)
    ax.set_thetagrids(angles * 180 / np.pi, labels)
    ax.set_rlabel_position(0.20)
    ax.set_title("订购服务雷达图")
    plt.show()
    return


def hq_user_payment_bar(col, condition):
    hq_df = consumer_Churn[(consumer_Churn[col] == condition) & (consumer_Churn['tenure'] > 24)
                           & (consumer_Churn['Churn'] == 0) & (consumer_Churn['MonthlyCharges'] > 60)].sort_values('tenure',ascending=False)
    for i, j in enumerate(['Contract', 'PaperlessBilling', 'PaymentMethod']):
        ax1 = fig.add_subplot(1, 3, i+1)
        ax1 = sns.countplot(x='Churn', hue=j, data=hq_df)
        ax1.set_title(j, fontsize=20)
    plt.show()
    return
# hq_user_payment_bar('SeniorCitizen', 1)
# high_quality_service('SeniorCitizen', 1)
# 老年高质量用户使用电话服务，多线服务，光纤，在线备份服务，设备保护服务，流媒体电视和电影，采用按月付费，无纸账单，电子支付方式
# high_quality_service('Partner', 'No')
# hq_user_payment_bar('Partner', 'No')
# 单身用户使用电话服务，多线服务，光纤，在线备份服务，设备保护服务，流媒体电视和电影，采用按月付费，无纸账单，非mail支付方式
# high_quality_service('Dependents', 'No')
# hq_user_payment_bar('Dependents', 'No')
# 无亲属用户使用电话服务，多线服务，光纤，在线备份服务，设备保护服务，流媒体电视和电影，采用两年支付方式，无纸账单，非mail支付方式
# high_quality_service('Contract','Month-to-month')
# hq_user_payment_bar('Contract','Month-to-month')
# 按月支付用户使用电话服务，光纤，多线服务，采用无纸账单，多采用电子支付

# 光纤服务的用户流失率过高，猜测用户在订购光纤之后在订购其他服务会降低流失率，所以找出流失率最低的组合


# add_service = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
# lst = []
# for i in ['Fiber optic']:
#     for j in add_service:
#         df_temp = consumer_Churn[(consumer_Churn['InternetService'] == i)]
#         for k in add_service[add_service.index(j):]:
#             df_temp = df_temp[(df_temp[k] == "Yes")]
#             unchurn_per = round(df_temp[(df_temp['Churn'] == 1)].shape[0]/df_temp.shape[0],3)
#             lst.append([i+" + "+" + ".join(add_service[add_service.index(j):][:add_service.index(k)+1]),unchurn_per])
# service_churn = pd.DataFrame(lst, columns=['产品套餐', '流失率']).sort_values('流失率', ascending=True)
# # print(service_churn)
# ax = fig.add_subplot(111)
# ax = sns.barplot(x='流失率', y='产品套餐', data=service_churn)
# ax.tick_params(labelsize=10)
# ax.set_xlabel(xlabel='流失率', fontsize=10)
# ax.set_ylabel(ylabel='产品套餐', fontsize=10)
# ax.axvline(x=Churn_p, ls=":", c="red")
# plt.show()

# 构建流失用户分类器，
# 特征处理
# consumer_Churn['gender'].replace({"Female": 0, "Male": 1}, inplace=True)
# consumer_Churn['Partner'].replace({"No": 0, "Yes": 1}, inplace=True)
# consumer_Churn['Dependents'].replace({"No": 0, "Yes": 1}, inplace=True)
# consumer_Churn['PhoneService'].replace({"No": 0, "Yes": 1}, inplace=True)
# consumer_Churn['MultipleLines'].replace({"No": 0, "Yes": 1, "No phone service": 2}, inplace=True)
# consumer_Churn['InternetService'].replace({"No": 0, "Fiber optic": 1, "DSL": 2}, inplace=True)
# consumer_Churn['OnlineSecurity'].replace({"No": 0, "Yes": 1, "No internet service": 2}, inplace=True)
# consumer_Churn['OnlineBackup'].replace({"No": 0, "Yes": 1, "No internet service": 2}, inplace=True)
# consumer_Churn['DeviceProtection'].replace({"No": 0, "Yes": 1, "No internet service": 2}, inplace=True)
# consumer_Churn['TechSupport'].replace({"No": 0, "Yes": 1, "No internet service": 2}, inplace=True)
# consumer_Churn['StreamingTV'].replace({"No": 0, "Yes": 1, "No internet service": 2}, inplace=True)
# consumer_Churn['StreamingMovies'].replace({"No": 0, "Yes": 1, "No internet service": 2}, inplace=True)
# consumer_Churn['Contract'].replace({"Month-to-month": 0, "Two year": 1, "One year": 2}, inplace=True)
# consumer_Churn['PaperlessBilling'].replace({"No": 0, "Yes": 1}, inplace=True)
# consumer_Churn['PaymentMethod'].replace({"Electronic check": 0, "Mailed check": 1, "Bank transfer (automatic)": 2, "Credit card (automatic)": 3}, inplace=True)
# # 归一化
# consumer_Churn['MonthlyCharges_scaler'] = (consumer_Churn['MonthlyCharges'])/(consumer_Churn['MonthlyCharges'].max()-consumer_Churn['MonthlyCharges'].min())
# consumer_Churn['TotalCharges_scaler'] = (consumer_Churn['TotalCharges'])/(consumer_Churn['TotalCharges'].max()-consumer_Churn['TotalCharges'].min())
# consumer_Churn['tenure_scaler'] = consumer_Churn['tenure']/(consumer_Churn['tenure'].max()-consumer_Churn['tenure'].min())
# consumer_Churn['tenure_bin'] = pd.cut(consumer_Churn['tenure_scaler'].values, bins=5, labels=[1, 2, 3, 4, 5])
# consumer_Churn['TotalCharges_bin'] = pd.cut(consumer_Churn['TotalCharges_scaler'].values, bins=5, labels=[1, 2, 3, 4, 5])
# consumer_Churn['MonthlyCharges_bin'] = pd.cut(consumer_Churn['MonthlyCharges_scaler'].values, bins=5, labels=[1, 2, 3, 4, 5])
# # 分离训练集预测试集
# from sklearn.model_selection import train_test_split
# train_data = consumer_Churn.drop_duplicates(subset=['customerID'],keep=False)
#
# data_columns = list(consumer_Churn.columns)
# data_columns.remove('customerID')
# data_columns.remove('Churn')
# data_columns.remove('MonthlyCharges')
# data_columns.remove('TotalCharges')
# data_columns.remove('tenure')
# data_columns.remove('tenure_scaler')
# data_columns.remove('MonthlyCharges_scaler')
# data_columns.remove('TotalCharges_scaler')
# target = train_data['Churn']
# # X_train, X_test, Y_train, Y_test = train_test_split(train_data[data_columns], target, test_size=0.2, random_state=0)
#
# # 选取特征值
# from sklearn.feature_selection import SelectKBest
# skb = SelectKBest(k=7)
# data_ = skb.fit_transform(train_data[data_columns], target)
# X_trainval, X_test, y_trainval, y_test = train_test_split(data_,target,test_size=0.2, random_state=0)
#
# from sklearn import svm
# from sklearn.naive_bayes import GaussianNB
# from sklearn.metrics import accuracy_score, recall_score
# # clf = svm.SVC(kernel='sigmoid')
# # clf.fit(X_trainval, y_trainval)
# gnb_clf = GaussianNB()
# gnb_clf.fit(X_trainval,y_trainval)
# y_predict = gnb_clf.predict(X_test)
# acc = accuracy_score(y_test, y_predict)
# rec = recall_score(y_test, y_predict)
# print('-----测试集----')
# print("准确率为%s" % acc)
# print("召回率为%s" % rec)
# vali_data = consumer_Churn.sample(frac=0.1)
# vali_data_ = skb.fit_transform(vali_data[data_columns], vali_data['Churn'])
# y_vali = gnb_clf.predict(vali_data_)
# acc2 = accuracy_score(vali_data['Churn'], y_vali)
# rec2 = recall_score(vali_data['Churn'], y_vali)
# print('-----验证集----')
# print("准确率为%s" % acc2)
# print("召回率为%s" % rec2)

index = ['一般价值客户', '一般保持客户', '一般发展客户', '一般挽留客户', '重要价值客户', '重要保持客户', '重要发展客户', '重要挽留客户']
value = [7181.28, 19937.45, 196971.23, 438291.81, 167080.83, 1592039.62, 45785.01, 33028.40]
data = pd.Series(index=index, data=value)
# print(data)
ax = fig.add_subplot(111)
ax.pie(x=data, labels=data.index, autopct='%3.2f%%')
plt.show()
