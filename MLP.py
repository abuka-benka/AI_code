import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
def sigmoid(x):
    return 1.0/(1+np.exp(-x))   #当输入是一个数组时，np会自动以元素为单位进行计算
class network(object):
    def __init__(self,layer):
        self.size=len(layer)
        self.layer=layer
        self.b=[np.random.randn(i,1) for i in layer[1:]]
        self.w=[np.random.randn(j,i) for i,j in zip(layer[:-1],layer[1:])]
    def MSE(self,ans,a):     #计算损失函数
        a=a.flatten()
        res=0
        for num,i in enumerate(a) :
            if num==ans:
                res+=(1-i)**2
            else:
                res+=i**2
        return res
    def fp(self,test):  #forward propagation
        res=0
        mse=0
        for a,ans in test:
            a=a.reshape(-1, 1)
            for b,w in zip(self.b,self.w):#矩阵计算下一层a
                a=sigmoid(np.dot(w,a)+b)
            if ans==np.argmax(a):#判断是否与答案相等
                res+=1
            mse+=self.MSE(ans,a)
        return (res,mse)
    def upd(self,TEST,eta,siz):
        delw=[np.zeros_like(w) for w in self.w]
        delb=[np.zeros_like(b) for b in self.b]
        for a,ans in TEST:
            a=a.reshape(-1, 1)
            val=[a]
            y=np.zeros((10,1))
            y[ans]=1
            for b,w in zip(self.b, self.w):
                a=sigmoid(np.dot(w,a)+b)
                val.append(a)
            for i in range(128):
                sum=0
                for k in range(10):
                    sum+=2*val[2][k][0]*self.w[1][k][i]*(1-val[2][k][0])*(val[2][k][0]-y[k][0])
                delb[0][i][0]+=eta*val[1][i][0]*(1-val[1][i][0])*sum
                for j in range(784):
                    delw[0][i][j]+=-eta*val[0][j][0]*val[1][i][0]*(1-val[1][i][0])*sum
            delw[1]-=np.dot((2*eta*val[2]*(1-val[2])*(val[2]-y)),val[1].T)
            delb[1]+=2*eta*val[2]*(val[2]-y)*(1-val[2])
        for i in range(len(self.w)):
            self.w[i]+=delw[i]/siz
        for i in range(len(self.b)):
            self.b[i]+=delb[i]/siz
    def deal(self,que,ans,T,siz,eta,x_t,y_t):
        TRAIN=list(zip(que,ans))
        TEST=list(zip(x_t,y_t))
        n=len(TRAIN)
        n_t=len(TEST)
        for j in range(T):
            random.shuffle(TRAIN)
            batch=[TRAIN[i:i+siz] for i in range(0,n,siz)]
            print(batch.shape)
            for i,mini in enumerate(batch):
                self.upd(mini,eta,siz)
            print ("round {0} : {1} / {2}".format(j+1,self.fp(TEST)[0],n_t))

mnist = fetch_openml('mnist_784', version=1, as_frame=False)
print("下载完成！")
x = mnist.data.astype('float32')/255.0
y = mnist.target.astype('int32')
x_train, x_test = x[:100], x[60000:]
y_train, y_test = y[:100], y[60000:]
net=network([784,128,10])
net.deal(x_train,y_train,35,10,1,x_train,y_train)