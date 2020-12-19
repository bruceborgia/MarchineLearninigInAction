perception
我们选取激活函数为：

![激活函数](https://github.com/bruceborgia/MarchineLearninigInAction/blob/master/ReadMeImagine/%E6%BF%80%E6%B4%BB%E5%87%BD%E6%95%B0.png)

这样就可以将线性回归的结果映射到两分类的结果上了。
定义损失函数为错误分类的数目，比较直观的方式是使用指示函数，但是指示函数不可导，因此可以定义：

![1](https://github.com/bruceborgia/MarchineLearninigInAction/blob/master/ReadMeImagine/%E6%8C%87%E7%A4%BA%E5%87%BD%E6%95%B0.png)

其中，是错误分类集合，实际在每一次训练的时候，我们采用梯度下降的算法。损失函数对  的偏导为：
但是如果样本非常多的情况下，计算复杂度较高，但是，实际上我们并不需要绝对的损失函数下降的方向，我们只需要损失函数的期望值下降，但是计算期望需要知道真实的概率分布，我们实际只能根据训练数据抽样来估算这个概率分布（经验风险）：

![](https://github.com/bruceborgia/MarchineLearninigInAction/blob/master/ReadMeImagine/%E6%A6%82%E7%8E%87%E5%88%86%E5%B8%83.png)

我们知道，  越大，样本近似真实分布越准确，但是对于一个标准差为  的数据，可以确定的标准差仅和  成反比，而计算速度却和  成正比。因此可以每次使用较少样本，则在数学期望的意义上损失降低的同时，有可以提高计算速度，如果每次只使用一个错误样本，我们有下面的更新策略（根据泰勒公式，在负方向）：

https://github.com/bruceborgia/MarchineLearninigInAction/blob/master/ReadMeImagine/%E8%BF%AD%E4%BB%A3%E6%96%B9%E7%A8%8B.png

是可以收敛的，同时使用单个观测更新也可以在一定程度上增加不确定度，从而减轻陷入局部最小的可能。在更大规模的数据上，常用的是小批量随机梯度下降法。
