import numpy as np
import matplotlib.pyplot as plt

p_x=np.array([[3,3],[4,3],[1,1]])
y=np.array([1,1,-1])
plt.figure()
for i in range(len(p_x)):
    if y[i]==1:
        plt.plot(p_x[i][0],p_x[i][1],'ro')#red point
    else:
        plt.plot(p_x[i][0],p_x[i][1],'bo')#blue point
# plt.show()

w=np.array([1,0])#初始化w
b=0#初始化截距 b
delta=1

for i in range(100):
    choice=-1
    for j in range(len(p_x)):
        if y[j]!=np.sign(np.dot(w,p_x[0])+ b):#sign为符号函数,sign(x),x>0 return 1;x<0 return -1,
                                                #dot函数为求积 矩阵内积
            choice=j
            break
    if choice==-1:
        break
    w=w+delta*y[choice]*p_x[choice]
    b=b+delta*y[choice]

line_x=[0,10]
line_y=[0,0]

for i in range(len(line_x)):
    line_y[i]=(-w[0]*line_x[i]-b)/w[1]

plt.plot(line_x,line_y)
plt.show()






# import numpy as np
# import matplotlib.pyplot as plt


# class Perceptron():

#     def __init__(self,alpha=0.01,n_iter=20):
#         self.alpha=alpha
#         self.n_iter=n_iter

#     def fit(self,X,Y):  
#         m,n=np.shape(X)
#         self.intercept=0
#         self.W=np.mat(np.ones((n,1)))

#         for i in range(self.n_iter):
#             for x,y in zip(X,Y):
#                 y_=float(x*self.W+self.intercept)
#                 if y*y_<0:
#                     self.W=x.T*self.alpha*y
#                     self.intercept+=self.alpha*y
#             if self.loss(X,Y)==0:
#                 return i

#     def loss(self,X,Y):
#         sum_loss=0.0
#         for x,y in zip(X,Y):
#             y_=float(x*self.W)+self.intercept
#             if(y*y_)<0:#<0表示分类错误
#                 sum_loss+=y_
#         return sum_loss

# if __name__ == "__main__":
#     X = np.mat(
#             np.array([[2, 4.3], [1, 2.4], [1, 3.3], [2, 1.8], [3, 9.2], [4, 6.3], [5, 10.1], [2.3, 3.4], [3.2, 6]]))
#     Y = np.array([1, 1, 1, -1, 1, -1, 1, -1, -1])
#     icons = {1: 'ro', -1: 'bo'}
#     # print X
#     for x, y in zip(X, Y):
#         plt.plot(x[0, 0], x[0, 1], icons[y])

#     perceptron = Perceptron(alpha=0.1, n_iter=20)

#     print(perceptron.fit(X, Y))

#     # points on the hyperplane
#     p1 = [0, -perceptron.intercept / perceptron.W[1, 0]]

#     p2 = [
#         5, (-perceptron.intercept - 5 * perceptron.W[0, 0]) / perceptron.W[1, 0]]

#     # print the hyperplane
#     plt.plot([p1[0], p2[0]], [p1[1], p2[1]])

#     plt.show()
