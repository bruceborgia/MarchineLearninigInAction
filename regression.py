import numpy as np

np.set_printoptions(precision=6)#解决终端中输出省略号
np.set_printoptions(threshold=np.inf) #控制台输出所有的值，不需要省略号

def loadDataSet(filename):#函数打开一个用tab键分开的文本文件,默认每行最后一个值为目标值
    numFeat=len(open(filename).readline().split('\t'))-1#得到数据的行数,以换行符为标识
    dataMat=[]
    labelMat=[]
    fr=open(filename)
    for line in fr.readlines():
        lineArr=[]
        curLine=line.strip().split('\t')
        for i in range(numFeat):
            lineArr.append(lineArr)
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat,labelMat

def standRegres(xArr,yArr):#计算最佳拟合直线
    xMat=np.mat(xArr)
    yMat=np.mat(yArr)
    xTx=xMat.T*xMat #x矩阵的转置xX矩阵
    if np.linalg.det(xTx)==0.0:#linalg为numpy中一个线性代数库,linalg.det(.)计算行列式,行列式非0则可逆
        print("this matrix is singular,cannot do inverse")
        return
    ws=xTx.I*(xMat.T*yMat)
    # ws=linalg.solve(xTx,xMat.T*yMat.T)
    return ws

if __name__ == "__main__":
    xArr,yArr=loadDataSet('8预测数值型数据回归\ex0.txt')
    
    #print(xArr)

    ws=standRegres(xArr,yArr)

    
    print(ws)

