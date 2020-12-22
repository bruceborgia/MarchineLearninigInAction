'''
    瓜数据集中一个西瓜属性有：色泽、根蒂、敲声、纹理、脐部和触感。
    对于色泽，有：青绿、乌黑、浅白，分别记为：1，0，-1。
    对于根蒂，有：蜷缩、稍蜷、硬挺，分别记为：-1，0，1。
    对于敲声，有：浊响、沉闷、清脆，分别记为：1，0，-1。
    对于纹理，有：清晰、稍糊、模糊，分别记为：1，0，-1。
    对于脐部，有：凹陷、稍凹、平坦，分别记为：1,0，-1。
    对于触感，有：硬滑、软粘，分别记为：1，-1。
    对于结果，有：好瓜记为1，坏瓜记为0。
'''

'''
对于使用的神经网络结构，设计为三层神经网络：输入层input、隐层hidden、输出层output。
输入层有七个神经元，为六个属性和一个偏置值（取1）；
隐层有五个神经元，为四个第一层的输出值和一个偏置值（取1）；
输出层有一个神经元
'''
import numpy as np
import matplotlib.pyplot as plt

data = [1,-1,1,1,1,0,1,
0,-1,0,1,1,0,1,
0,-1,1,1,1,0,1,
1,-1,0,1,1,0,1,
1,-1,1,1,1,0,1,
1,0,1,1,0,-1,1,
0,0,1,1,0,-1,1,
0,0,1,0,0,0,1,
0,0,0,1,0,0,1,
1,1,-1,0,-1,-1,1,
-1,1,-1,-1,-1,0,1,
-1,-1,1,-1,-1,-1,1,
1,0,1,0,1,0,1,
-1,0,0,0,1,0,1,
0,0,1,1,0,-1,1,
-1,-1,1,-1,-1,0,1,
0,-1,0,0,0,0,1]

data_set = np.array(data).reshape(17,7)# X输入值
results = [1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0]
y = np.array(results).reshape(17,1)#y输入值 y^

w1_v_ih=[]#v_ih
w2_w_hj=[]#w_hj

np.random.seed(1)#设定随机种子数

#随机产生输入层到隐层的权值
for i in range(7):#7x4
    for j in range(1,5):
        w=np.random.randn()
        w1_v_ih.append(w)

#随机产生隐层到输出层的权值
for d in range(1,6): #5x1
    w=np.random.randn()
    w2_w_hj.append(w)
    
#将权重转换成矩阵
w_IH=np.array(w1_v_ih).reshape(7,4)#IN-Hidden
w_HO=np.array(w2_w_hj).reshape(5,1)#Hidden-Out

a2=[]#迭代次数
b2=[]#损失函数
i=0
#取学习率为0.2
learning_rate=0.2


while True:
    #前向传播
    alpha=np.dot(data_set,w_IH)#x乘以权重w,得到隐层的输入值
    b=1/(1+np.exp(-alpha))#sigmoid 函数 隐层激活函数
    b1=np.c_[b,np.ones(17)]
    '''
        np.c_ 用于连接两个矩阵
        x_1 = 
            [[1 2 3]
            [4 5 6]]
        x_2 = 
            [[3 2 1]
            [8 9 6]]
        
        x_new = np.c_[x_1,x_2]
        x_new = 
            [[1 2 3 3 2 1]
            [4 5 6 8 9 6]]

        np.ones(17) 返回全是1的np.array()数组

    '''
    beta=np.dot(b1,w_HO)#得到输出层的输入层
    y_j = 1/(1 + np.exp(-beta))#sigmoid 函数 输出层激活函数,y_j为最终输出
    # print(y_j)
    #计算误差
    err=(np.dot(np.transpose(y-y_j),(y-y_j))/(17*2))#np.transpose()为转置
    print(err)
    i+=1#记录迭代次数
    a2.append(i)#写入列表count
    b2.append(float(err))#记录代价函数
    

    #更新权重退出条件,误差<0.015
    if err<0.015:
        break
    
    #如果误差不满足条件,则反向传播更新权重
    else:
        w_IH += learning_rate*np.dot(np.transpose(data_set),(np.dot(((y-y_j)*(y_j*(1-y_j))),(np.transpose(np.delete(w_HO,4,axis=0))))*(b*(1-b))))#更新输入层到隐层权重
        w_HO += np.array(learning_rate/17*(((y-y_j)*(y_j*(1-y_j))*(b1)).sum(axis=0))).reshape(5,1)#更新隐层到输出层权重
        '''
            y=np.array([[4,8],[5,6]])
            c=y.sum(axis=0)
            [ 9 14]
            c=y.sum(axis=1)
            [12 11]
        '''
	

print(w_IH)
print(w_HO)
print(y_j)

fig = plt.figure()
ax=fig.add_subplot(1,1,1)
plt.xlabel("generation(end when generation reach " + str(i) + ")")
plt.ylabel("err")
plt.plot(a2,b2)
plt.title("the err varies with generation(end when err<0.015)")
plt.show()


data_1 = []
for i in range(0,17):
	for j in range(i*7,(i+1)*7):
		data_1.append(data[j])
	data_set1 = np.array(data_1).reshape(1,7)
	z1 = np.dot(data_set1,w_IH)
	f1 = 1/(1 + np.exp(-z1))
	f = np.c_[f1,np.ones(1)]
	z2 = np.dot(f,w_HO)
	o = 1/(1 + np.exp(-z2))
    
	print(data_set1)
	if o <= 0.5:
		print("the predicted value is " + str(float(o)),"\n","This is a bad watermelon")
	else:
		print("the pridicted value is "+str(float(o)),"\n","This is a good watermelon")
	data_1 = []
	print()

	