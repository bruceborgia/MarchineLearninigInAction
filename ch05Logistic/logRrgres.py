from os import error
import numpy as np
from numpy.core.numeric import ones
from numpy.ma import exp
import matplotlib.pyplot as plt
from numpy.ma.core import array

def loadDataSet():
    dataMat=[]#x1,x2
    labelMat=[]#y
    fr=open('ch05Logistic\\testSet.txt')#打开文件
    for line in fr.readlines():#逐行读取
        lineArr=line.strip().split()#去掉每行两边的空白字符,并以空格分隔每行数据元素
        dataMat.append([1.0,float(lineArr[0]),float(lineArr[1])])#将x0设为1,x1,x2为从txt中读取,因此x特征为为n*3,其中第一位恒为1
        labelMat.append(int(lineArr[2]))#y
    return dataMat,labelMat#labelMat是1*100行向量

def sigmoid(inX):
    return 1.0/(1+exp(-inX))

def gradAscent(dataMatIn,classLables):#梯度上升学习法,和梯度下降的区别是迭代时上升为+,下降为-
    #dataMatIn为3维numpy数组,代表不同的特征,每行代表一个训练样本 100*3维
    #classLables为标签y,是1*100的行向量
    dataMatrix=np.mat(dataMatIn)#转换成numpy矩阵
    labelMat=np.mat(classLables).transpose()#转换成numpy矩阵且由原来的1*100行向量转换成100*1的列向量
    m,n=np.shape(dataMatrix)#x的维数是mxn

    alpha=0.001#学习率 α
    maxCycles=500#最大迭代次数
    weights=np.ones((n,1))
    for k in range(maxCycles):
        h=sigmoid(dataMatrix*weights)#h是100*1的列向量
        error=(labelMat-h)
        weights=weights+alpha*dataMatrix.transpose()*error
    return weights

def plotBestFit(weights):#画出数据集和logistic回归的最佳拟合直线

    dataMat,labelMat=loadDataSet()

    #转换成numpy的array数组
    dataArr=np.array(dataMat)

    # 数据个数
    # 例如建立一个4*2的矩阵c，c.shape[1]为第一维的长度2， c.shape[0]为第二维的长度4
    n=np.shape(dataArr)[0]

    #正例
    xcord1=[]
    ycord1=[]

    #反例
    xcord2=[]
    ycord2=[]

    #根据数据集标签进行分类
    for i in range(n):
        if int(labelMat[i])==1:#正例
            xcord1.append(dataArr[i,1])
            ycord1.append(dataArr[i,2])
        else:#反例
            xcord2.append(dataArr[i,1])
            ycord2.append(dataArr[i,2])

    fig=plt.figure()
    ax=fig.add_subplot(111)
    ax.scatter(xcord1,ycord1,s=30,c='red',marker='s',alpha=.5)
    ax.scatter(xcord2,ycord2,s=30,c='green',alpha=.5)
    #x轴坐标
    x=np.arange(-3.0,3.0,0.1)#创建等差数组，-3为起点，3为重点，0.1为步长
    # w0*x0 + w1*x1 * w2*x2 = 0
    # x0 = 1, x1 = x, x2 = y
    y=(-weights[0]-weights[1]*x)/weights[2]
    ax.plot(x,y)
    plt.title('bestfit')
    plt.xlabel('X1')
    plt.ylabel('x2')
    plt.show()

def stocGradAscent0(dataMatrix,classLabels):
    """
    随机梯度上升
    """
    m,n=np.shape(dataMatrix)
    #固定学习率
    alpha=0.01
    weights=ones(n)
    for i in range(m):
        #梯度上升矢量公式
        h=sigmoid(sum(dataMatrix[i]*weights))#向量
        error=classLabels[i]-h#向量
        weights=weights+alpha*error*dataMatrix[i]
        # numpy.append(arr, values, axis=None):就是arr和values会重新组合成一个新的数组，做为返回值。
        # 当axis无定义时，是横向加成，返回总是为一维数组
    return weights

def stocGradAscent1(dataMatrix,classLabels,numIter=150):
    """
    改进的随机梯度上升
    """
    m,n=np.shape(dataMatrix)
    weights=ones(n)
    for j in range(m):
        dataIndex=list(range(m))
        for i in range(m):
            #每次都降低alpha的大小
            alpha=4/(1.0+j+i)+0.01
            #随机选择样本
            randIndex=int(np.random.uniform(0,len(dataIndex)))
            #随机选择一个样本计算h
            h=sigmoid(sum(dataMatrix[randIndex]*weights))
            #计算误差
            error=classLabels[randIndex]-h
            #更新回归系数
            weights=weights+alpha*error*dataMatrix[randIndex]
            #删除已使用版本
            del(dataIndex[randIndex])
    return weights


if __name__ == "__main__":
    dataArr,labelMat=loadDataSet()
    weights=gradAscent(dataArr,labelMat)
    print(weights)

    plotBestFit(weights.getA())

    weights=stocGradAscent0(array(dataArr),labelMat)
    print(weights)
    plotBestFit(weights)

    weights=stocGradAscent1(array(dataArr),labelMat,500)
    print(weights)
    plotBestFit(weights)
