#### 微博情感分析
---------
学者在文本挖掘挖掘展开了很多的研究，微博，作为一个国内较为火热的社交平台，其涉及的内容十分广发，如娱乐、影视、体育等，不同的内容有着不同的研究，本文结合课上和课后所学知识，运用现在的微博情感分析方法，开展本次实验。实验表明LinearSVM和Bi-LSTM的效果比较好，准确率达到96%以上。

对于情感分类，早期的方法是基于支持向量机或统计模型，而最近的方法采用了递归神经网络等深度学习的方法。一个文本序列通过Word2Vec等词嵌入(word embedding)模型转化为词向量(word vector)序列作为模型的输入，使特征具有语义信息，深度学习构建网络模型模拟人脑神经系统对文本进行逐步分析、特征抽取并且自动学习优化模型输出，以提高分类的准确性[9]。情感分类任务可以划分为两大类：(1)传统深度学习的方法，典型代表CNN、LSTM和RNN等；(2)图神经网络的方法，典型代表有GCN等。

虽然LinearSVM比较简单，但是调优LinearSVM的训练这个过程是相当有启发性的事情，LinearSVM每次只取使得损失函数极大的一个样本进行梯度下降，这就导致模型在某个地方可能来来回回都只受那么几个样本的影响，加上对测试集的未知性，最终还是采用Bi-LSTM进行情感分类。



####  结果与分析

特征提取的方法有很多，因此我们进行实验来探寻一个较为好的方法，如下图7所示，可以看出，当jieba_feature特征提取和词袋特征提取效果，在6个情感分类器中都比较稳定。



图7 词特征对分类器的影响。横坐标表示分别是BernoulliNB classifier、MultinomiaNB classifier、LogisticRegression classifier、SVC classifier、LinearSVC classifier、NuSVC classifier，这些分类器在bigram、double bigram、jieba feature extraction and bag of words下的准确率

 ![result2](https://github.com/HappyZXY/weibo-reviews-sentiment-analysis/blob/main/res/pic/result2.png)

图8 特征提取与分类器之间的关系

进一步探究特征提取器与分类器的关系，如上图8所示，可以发现LinearSV-M、SVC和线性回归方法的效果比较好，准确率达到95%左右，F-Score值亦是这三个分类器表现较为出类拔萃。

下图9是Bi-LSTM模型在微博评论数据集上的分类效果，实验中总共用了6次迭代，可以看出模型的准确率已经很高了，模型损失也在第2、3次迭代后快速下降，模型在训练集和验证集上都比较稳定，一方面的原因在于本身数据标签比较精准，使得模型“学习效率比较高”。

![result3](https://github.com/HappyZXY/weibo-reviews-sentiment-analysis/blob/main/res/pic/result3.png)

图9 Bi-LSTM情感分类器效果(左边为模型的准确率，右边为模型的损失)



#### Code

* code（实验代码）<br>
  * earlyMethodsProPre.py<br>
  * testEarlyMethods.py：对BernoulliNB、MultinomiaNB、LogisticRegression、SVC、NuSVM、LinearSVC分类器进行实验。<br>
  * sub_obj.py:	主观句和客观句分类。<br>


* dataset（数据集）<br>
  * 情感词和停用词表<br>
  * 正负样例：pos.txt   neg.txt<br>
