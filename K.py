
#引入依赖
import  numpy as np
import pandas as pd

#数据的加载和预处理,这里直接引入sklearn的数据集
from sklearn.datasets import load_iris #导入鸢尾花数据集
from sklearn.model_selection import  train_test_split #切分数据集与训练集与数据集
from sklearn.metrics import  accuracy_score #计算分类预测准确度的

iris = load_iris() #加载数据集,data数据相当于x,target相当于y，取得离散值0，1，2表示每个样本点对应的类别,
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['class'] = iris.target #增加一列类别class
#将对应的0，1，2改成对应的分类的花名
df['class'] = df['class'].map({0:iris.target_names[0], 1:iris.target_names[1], 2:iris.target_names[2]})
x = iris.data
y = iris.target.reshape(-1,1) #转换成二维列向量

#划分训练集与测试集
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=35, stratify=y) #测试集占比30%，按照y等比例分层，x随机

#核心算法实现
#定义距离函数
def l1_distance(a, b):
    return np.sum(np.abs(a-b), axis=1)
def l2_distance(a,b):
    return np.sqrt(np.sum((a-b)**2,axis=1))

#分类器的实现
class KNN(object):
    #定义类的构造方法
    def __init__(self, n_neighbors=1, dis_fun=l1_distance):
        self.n_neighbors = n_neighbors
        self.dis_fun = dis_fun

    #训练模型的方法
    def fit(self, x, y):
        self.x_train =  x
        self.y_train = y

    #预测模型的方法
    def predict(self, x):
        #初始化预测分类数组
        y_pred = np.zeros((x.shape[0],1), dtype=self.y_train.dtype)
        #遍历输入的x个测试点,取出每个数据点的序号i和数据x_test
        for i, x_test in enumerate(x):
            #1.x_test要跟所有训练数据计算距离
            distance = self.dis_fun(self.x_train, x_test)
            #2.得到的距离按照由近到远排序,从近到远对应的索引
            nn_index = np.argsort(distance)
            #3.选取距离最近的k个点,保存它们的对应类别
            nn_y = self.y_train[nn_index[:self.n_neighbors]].ravel()
            #4.统计类别中出现频率最高的，赋给y_pred
            y_pred[i] = np.argmax(np.bincount(nn_y))
        return y_pred

#测试

'''knn = KNN(n_neighbors=3)
#训练模型
knn.fit(x_train, y_train)
#模型预测
y_pred = knn.predict(x_test)
#求出预测准确率
accuracy = accuracy_score(y_test, y_pred)
print("预测准确率:", accuracy)'''

knn = KNN()
#训练模型
knn.fit(x_train, y_train)
#保存结果
result_list = []
#针对不同的参数选取,做预测,两个不同的距离计算
for p in [1, 2]:
    knn.dis_fun = l1_distance if p==1 else l2_distance
    #考虑到不同的k值，尽量选奇数,所以步长为2
    for k in range(1,10,2):
        knn.n_neighbors = k
        #预测
        y_pred = knn.predict(x_test)
        #计算准确率
        accuracy = accuracy_score(y_test, y_pred)
        result_list.append([k, 'l1_distance' if p==1 else '2_distance', accuracy])
df = pd.DataFrame(result_list, columns=['k', '距离函数', '准确率'])
print(df)







