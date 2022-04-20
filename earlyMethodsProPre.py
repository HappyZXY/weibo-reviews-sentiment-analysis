import sys
import pandas as pd
import os
import jieba
import nltk
nltk.download('punkt')
from nltk.collocations import  BigramCollocationFinder
from nltk.metrics import  BigramAssocMeasures
from nltk.probability import  FreqDist,ConditionalFreqDist
from nltk.metrics import  BigramAssocMeasures
from random import shuffle
import sklearn
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score as AUC
os.environ["export CUDA_VISIBLE_DEVICES"] = '0'

class dataPrePro():
    # 获取所有文本	
    def text(self, save=None, pos_save=None, neg_save=None):
        filePath = os.path.join('dataset/data',"train.csv" )
        df = pd.read_csv(filePath)
        list1 = df['review'].tolist()
        labels = df['label'].tolist()
        # print(int(labels[0])) #0
        # print(len(list1))
        reviews = ""
        for (review, label) in zip(list1, labels):
            reviews = reviews+review
            if save==True:
                with open("dataset/totalReviews.txt",'a+', encoding="utf-8") as f1:
                    f1.writelines(review)
                    f1.write('\n')
            if pos_save==True and int(label) == 1:
                with open("dataset/posReviews.txt",'a+', encoding="utf-8") as f2:
                    f2.writelines(review)
                    f2.write('\n')
            if pos_save==True and int(label) == 0:
                with open("dataset/negReviews.txt",'a+', encoding="utf-8") as f3:
                    f3.writelines(review)
                    f3.write('\n')
        return reviews 




    # 把单个词作为特征
    def bag_of_words(self, words):
        return dict([(word,True) for word in words])

    def bigram(self, words,score_fn=BigramAssocMeasures.chi_sq,n=1000):
        bigram_finder=BigramCollocationFinder.from_words(words)#把文本变成双词搭配的形式
        # print("双词：{}".format(bigram_finder))
        bigrams = bigram_finder.nbest(score_fn,n)#使用卡方统计的方法，选择排名前1000的双词
        newBigrams = [u+v for (u,v) in bigrams]
        return self.bag_of_words(newBigrams)

    def bigram_words(self, words,score_fn=BigramAssocMeasures.chi_sq,n=1000):
        bigram_finder=BigramCollocationFinder.from_words(words)
        bigrams = bigram_finder.nbest(score_fn,n)
        newBigrams = [u+v for (u,v) in bigrams]
        a = self.bag_of_words(words)
        b = self.bag_of_words(newBigrams)
        a.update(b)#把字典b合并到字典a中
        return a#所有单个词和双个词一起作为特征

    def dropStopwords(self, filename):
        stop = [line.strip() for line in  open('dataset/dic/stop.txt','r').readlines()]#停用词
        f = open(filename,'r')
        line = f.readline()
        #line.decode('utf-8').encode('gbk', 'ignore')
        str1 = []
        while line:
            s = line.strip('\n')
            s = s.replace(' ','')
            fenci = jieba.cut(s,cut_all=False)#False默认值：精准模式
            str1.append(list(set(fenci)-set(stop)))	
            line = f.readline()
        return str1


    #获取信息量最高(前number个)的特征(卡方统计)
    def jieba_feature(self, number):
        '''
        number是特征的维度，是可以不断调整直至最优的
        '''
        posWords = []
        negWords = []
        for items in self.dropStopwords('dataset/posReviews.txt'):
            for item in items:
                posWords.append(item)
        
        for items in self.dropStopwords('dataset/negReviews.txt'):
            for item in items:
                negWords.append(item)
        
        #统计所有词的词频
        word_fd = FreqDist()
        # 统计消极评论和积极评论中的词频
        cond_word_fd = ConditionalFreqDist()
        for word in posWords:
            word_fd[word] += 1
            cond_word_fd['pos'][word] += 1
        for word in negWords:
            word_fd[word] += 1
            cond_word_fd['neg'][word] += 1
        pos_word_count = cond_word_fd['pos'].N() #积极词的数量
        # print("*pos_word_count:\n{}".format(pos_word_count))
        neg_word_count = cond_word_fd['neg'].N() #消极词的数量
        # print("*neg_word_count:\n{}\n".format(neg_word_count))
        total_word_count = pos_word_count + neg_word_count
        
        word_scores = {} #包括了每个词和这个词的信息量
        sentiment = []
        file1 = open("dataset/dic/sentiment.txt")
        while 1:
            line1 = file1.readline()
            line1 = line1.strip()
            line1 = line1.replace(' ', '')
            sentiment.append(line1)
            if not line1:
                break
        for word, freq in word_fd.items(): #统计所有词的词频 word_fd
            flag1=1
            if word in sentiment:
                flag1 = 0
            if flag1==1:
                #计算积极词的卡方统计量，这里也可以计算互信息等其它统计量
                pos_score = BigramAssocMeasures.chi_sq(cond_word_fd['pos'][word], (freq, pos_word_count), total_word_count)
                neg_score = BigramAssocMeasures.chi_sq(cond_word_fd['neg'][word],  (freq, neg_word_count), total_word_count) 
                word_scores[word] = pos_score + neg_score #一个词的信息量等于积极卡方统计量加上消极卡方统计量
            else: # 有情绪词
                pos_score = BigramAssocMeasures.chi_sq(cond_word_fd['pos'][word],  (freq, pos_word_count), total_word_count) #计算积极词的卡方统计量，这里也可以计算互信息等其它统计量
                neg_score = BigramAssocMeasures.chi_sq(cond_word_fd['neg'][word],  (freq, neg_word_count), total_word_count) #同理
                word_scores[word] = pos_score + neg_score + 0.4*2.26036810405 #一个词的信息量等于积极卡方统计量加上消极卡方统计量			
        #把词按信息量倒序排序。number是特征的维度，是可以不断调整直至最优的
        best_vals = sorted(word_scores.items(), key=lambda item:item[1],  reverse=True)[:number]
        best_words = set([w for w,s in best_vals])
        tmp = dict([(word, True) for word in best_words])
        return tmp

    
    def build_features(self):
        # print("======================jieba_feature:==========================")
        # feature = self.jieba_feature(1000)#结巴分词

        print("======================bigram:==========================")
        feature = self.bigram(words=self.text())
        
        # print("======================bigram_words:==========================")
        # feature = self.bigram_words(words=self.text())

        # print("======================bag_of_words:==========================")
        # feature = self.bag_of_words(words=self.text()) 

        posFeatures = []
        for items in self.dropStopwords('dataset/posReviews.txt'):
            a = {}
            for item in items:
                if item in feature.keys():
                    a[item]='True'
            posWords = [a,'pos'] #为积极文本赋予"pos"
            posFeatures.append(posWords)

        negFeatures = []
        for items in self.dropStopwords('dataset/negReviews.txt'):
            a = {}
            for item in items:
                if item in feature.keys():
                    a[item]='True'
            negWords = [a,'neg'] #为消极文本赋予"neg"
            negFeatures.append(negWords)
        return posFeatures,negFeatures

if __name__ == "__main__":
    test = dataPrePro()
    # test.text(save=True, pos_save=True, neg_save=True)