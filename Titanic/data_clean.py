# coding:utf-8
'''
Created on 2018/1/29.

@author: chk01
'''
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn import model_selection
import matplotlib.pyplot as plt

train_org_data = pd.read_csv('data/train.csv')
test_data = pd.read_csv('data/test.csv')
train_data = train_org_data.copy(deep=True)
data = [train_data, test_data]

# 读取数据-查看数据-了解每个字段含义
# 检查缺失空值
# print(train_data.info())
# 总共891个样本
isnull_table = train_data.isnull().sum()
sorted_isnull_table = isnull_table.sort_values(ascending=False)
# print(sorted_isnull_table)

# print(tt.sort_index(ascending=False))
# 1、Correcting
# 1、、常见方法：简单统计量分析-3原则-箱型图分析
# 2、Completing
# 3、Creating
# 4、Converting
# print(train_data.describe(include='all'))

# print(data.keys())
# ['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']

# 2Completing修复空值
for d in data:
    # 年龄和费用用中位数修复
    d['Age'].fillna(d['Age'].median(), inplace=True)
    d['Fare'].fillna(d['Fare'].median(), inplace=True)
    # print(d['Embarked'].mode())
    # 登入港口用次数最多的修复
    # mode方法返回出现次数最多的值（列表），次数相同时，返回多个
    d['Embarked'].fillna(d['Embarked'].mode()[0], inplace=True)

# 删除无用列-乘客编号，船舱，船票编码
drop_column = ['PassengerId', 'Cabin', 'Ticket']
train_data.drop(drop_column, axis=1, inplace=True)

# 3Creating组合新特征
for d in data:
    # Discrete variables
    d['FamilySize'] = d['SibSp'] + d['Parch'] + 1

    d['IsAlone'] = 1  # initialize to yes/1 is alone
    d['IsAlone'].loc[d['FamilySize'] > 1] = 0  # now update to no/0 if family size is greater than 1

    # # quick and dirty code split title from name: http://www.pythonforbeginners.com/dictionary/python-split
    d['Title'] = d['Name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]

    # 序列数据分区
    d['FareBin'] = pd.qcut(d['Fare'], 4)
    d['AgeBin'] = pd.cut(d['Age'].astype(int), 5)

stat_min = 10
# value_counts统计每个值出现的次数
title_names = (train_data['Title'].value_counts() < stat_min)
train_data['Title'] = train_data['Title'].apply(lambda x: 'Misc' if title_names[x] == True else x)

# 4 Convert 数据格式转化-将对象转化为编程01
label = LabelEncoder()
for d in data:
    # female:0-male:1
    d['Sex_Code'] = label.fit_transform(d['Sex'])
    d['Embarked_Code'] = label.fit_transform(d['Embarked'])
    d['Title_Code'] = label.fit_transform(d['Title'])
    d['AgeBin_Code'] = label.fit_transform(d['AgeBin'])
    d['FareBin_Code'] = label.fit_transform(d['FareBin'])

Target = ['Survived']

# define x variables for original features aka feature selection
data1_x = ['Sex', 'Pclass', 'Embarked', 'Title', 'SibSp', 'Parch', 'Age', 'Fare', 'FamilySize',
           'IsAlone']  # pretty name/values for charts
data1_x_calc = ['Sex_Code', 'Pclass', 'Embarked_Code', 'Title_Code', 'SibSp', 'Parch', 'Age',
                'Fare']  # coded for algorithm calculation
data1_xy = Target + data1_x
print('Original X Y: ', data1_xy, '\n')

# define x variables for original w/bin features to remove continuous variables
data1_x_bin = ['Sex_Code', 'Pclass', 'Embarked_Code', 'Title_Code', 'FamilySize', 'AgeBin_Code', 'FareBin_Code']
data1_xy_bin = Target + data1_x_bin
print('Bin X Y: ', data1_xy_bin, '\n')

# define x and y variables for dummy features original
data1_dummy = pd.get_dummies(train_data[data1_x])
data1_x_dummy = data1_dummy.columns.tolist()
data1_xy_dummy = Target + data1_x_dummy
print('Dummy X Y: ', data1_xy_dummy, '\n')

train1_x, test1_x, train1_y, test1_y = model_selection.train_test_split(train_data[data1_x_calc], train_data[Target],
                                                                        random_state=0)
train1_x_bin, test1_x_bin, train1_y_bin, test1_y_bin = model_selection.train_test_split(train_data[data1_x_bin],
                                                                                        train_data[Target],
                                                                                        random_state=0)
train1_x_dummy, test1_x_dummy, train1_y_dummy, test1_y_dummy = model_selection.train_test_split(
    data1_dummy[data1_x_dummy], train_data[Target], random_state=0)

print("Data1 Shape: {}".format(train_data.shape))
print("Train1 Shape: {}".format(train1_x.shape))
print("Test1 Shape: {}".format(test1_x.shape))

print(train1_x.shape)
for x in train1_x:
    if train_data[x].dtype != 'float64':
        print('Survival Correlation by:', x)
        print(train_data[[x, Target[0]]].groupby(x, as_index=False).mean())
        print('-' * 10, '\n')

data1 = train_data
plt.figure(figsize=[16, 12])

plt.subplot(231)
plt.boxplot(x=data1['Fare'], showmeans=True, meanline=True, whis=6.2)
plt.title('Fare Boxplot')
plt.ylabel('Fare ($)')

plt.subplot(232)
plt.boxplot(data1['Age'], showmeans=True, meanline=True, whis=2)
plt.title('Age Boxplot')
plt.ylabel('Age (Years)')

plt.subplot(233)
plt.boxplot(data1['FamilySize'], showmeans=True, meanline=True)
plt.title('Family Size Boxplot')
plt.ylabel('Family Size (#)')

plt.subplot(234)
plt.hist(x=[data1[data1['Survived'] == 1]['Fare'], data1[data1['Survived'] == 0]['Fare']],
         stacked=True, color=['g', 'r'], label=['Survived', 'Dead'])
plt.title('Fare Histogram by Survival')
plt.xlabel('Fare ($)')
plt.ylabel('# of Passengers')
plt.legend()

plt.subplot(235)
plt.hist(x=[data1[data1['Survived'] == 1]['Age'], data1[data1['Survived'] == 0]['Age']],
         stacked=True, color=['g', 'r'], label=['Survived', 'Dead'])
plt.title('Age Histogram by Survival')
plt.xlabel('Age (Years)')
plt.ylabel('# of Passengers')
plt.legend()

plt.subplot(236)
plt.hist(x=[data1[data1['Survived'] == 1]['FamilySize'], data1[data1['Survived'] == 0]['FamilySize']],
         stacked=True, color=['g', 'r'], label=['Survived', 'Dead'])
plt.title('Family Size Histogram by Survival')
plt.xlabel('Family Size (#)')
plt.ylabel('# of Passengers')
plt.legend()
plt.show()
