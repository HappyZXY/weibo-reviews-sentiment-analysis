#### 微博情感分析
 ---------
学者在文本挖掘挖掘展开了很多的研究，微博，作为一个国内较为火热的社交平台，其涉及的内容十分广发，如娱乐、影视、体育等，不同的内容有着不同的研究，本文结合课上和课后所学知识，运用现在的微博情感分析方法，开展本次实验。实验表明LinearSVM和Bi-LSTM的效果比较好，准确率达到96%以上。

对于情感分类，早期的方法是基于支持向量机或统计模型，而最近的方法采用了递归神经网络等深度学习的方法。一个文本序列通过Word2Vec等词嵌入(word embedding)模型转化为词向量(word vector)序列作为模型的输入，使特征具有语义信息，深度学习构建网络模型模拟人脑神经系统对文本进行逐步分析、特征抽取并且自动学习优化模型输出，以提高分类的准确性[9]。情感分类任务可以划分为两大类：(1)传统深度学习的方法，典型代表CNN、LSTM和RNN等；(2)图神经网络的方法，典型代表有GCN等。

虽然LinearSVM比较简单，但是调优LinearSVM的训练这个过程是相当有启发性的事情，LinearSVM每次只取使得损失函数极大的一个样本进行梯度下降，这就导致模型在某个地方可能来来回回都只受那么几个样本的影响，加上对测试集的未知性，最终还是采用Bi-LSTM进行情感分类。


#### Code

* code（实验代码）<br>
  * earlyMethodsProPre.py<br>
  * testEarlyMethods.py：对BernoulliNB、MultinomiaNB、LogisticRegression、SVC、NuSVM、LinearSVC分类器进行实验。<br>
  * sub_obj.py:	主观句和客观句分类。<br>


* dataset（数据集）<br>
  * 情感词和停用词表<br>
  * 正负样例：pos.txt   neg.txt<br>
