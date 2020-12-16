import numpy as np
import matplotlib.pyplot as plt


def loadDataSet(filename):#函数打开一个用tab键分开的文本文件,默认每行最后一个值为目标值
    numFeat=len(open(filename).readline().split('\t'))-1#得到数据的行数,以换行符为标识 特征数
    dataMat=[]
    labelMat=[]
    fr=open(filename)
    for line in fr.readlines():
        lineArr=[]
        curLine=line.strip().split('\t')
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat,labelMat

def standRegres(xArr,yArr):#计算回归系数w
    xMat=np.mat(xArr)
    yMat=np.mat(yArr).T
    xTx=xMat.T*xMat #x矩阵的转置xX矩阵
    if np.linalg.det(xTx)==0.0:#linalg为numpy中一个线性代数库,linalg.det(.)计算行列式,行列式非0则可逆
        print("this matrix is singular,cannot do inverse")
        return
    ws=xTx.I*(xMat.T*yMat)
    # ws=linalg.solve(xTx,xMat.T*yMat.T) 
    return ws

def pltTrueData():#画出原始数据的散点图
    xMat=np.mat(xArr)
    yMat=np.mat(yArr)
    fig=plt.figure()
    ax=fig.add_subplot(111)
    ax.scatter(xMat[:,1].flatten().A[0],yMat.T[:,0].flatten().A[0],s=20,c='blue',alpha=.5)
    plt.show()


def plotDataSet():#绘制原始数据散点图和最佳拟合直线图
    xArr,yArr=loadDataSet('8预测数值型数据回归\ex0.txt')
    ws=standRegres(xArr,yArr)#变量系数 y=ws[0]+ws[1]*x1

    xMat=np.mat(xArr)
    yMat=np.mat(yArr)

    xCopy=xMat.copy()
    xCopy.sort(0)#升序排列
    yHat=xCopy*ws#预测值的y 使用排序完后的x进行预测
    
    #计算预测值和真实值的相关性
    #corrcoef函数得到相关系数矩阵
    #得到的结果中对角线上的数据是1.0,因为yMat和自己的匹配是完美的,因此为1
    #而yMat1和yMat的相关系数为0.98
    yHat1=xMat*ws
    print(np.corrcoef(yHat1.T,yMat))#yMat需要转置,以保证两个向量都是行向量
    

    #绘制原始数据散点图和最佳拟合直线图
    fig=plt.figure()
    ax=fig.add_subplot(111)
    ax.plot(xCopy[:,1],yHat,c='red')# 绘制样本点,颜色为红色
    ax.scatter(xMat[:, 1].flatten().A[0], yMat.flatten().A[0], s=20, c='orange', alpha=0.5)#散点图,alpha为透明度,取值为0-1之间
    #flatten返回一个折叠成一维的数组.但该函数只能适用于numpy对象,即array或者mat,普通的list不行
    #矩阵.A(等效于矩阵.getA())变成了数组
    plt.title('dataset')
    plt.xlabel('X')
    plt.show()



if __name__ == "__main__":
    xArr,yArr=loadDataSet('8预测数值型数据回归\ex0.txt')

    ws=standRegres(xArr,yArr)#变量系数 y=ws[0]+ws[1]*x1
    print(ws)

    pltTrueData()

    plotDataSet()
   

    

    

