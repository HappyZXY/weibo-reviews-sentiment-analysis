from earlyMethodsProPre import dataPrePro
from random import shuffle
import os
from nltk.classify.scikitlearn import SklearnClassifier
import sklearn
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score as AUC
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
os.environ["export CUDA_VISIBLE_DEVICES"] = '0'



def score(classifier): 
    classifier = SklearnClassifier(classifier) #在nltk中使用scikit-learn的接口
    classifier.train(train) #训练分类器
    pred = classifier.classify_many(data)
    TN = 0
    m=0 #neg
    n=0 #pos
    TP = 0
    s = len(pred)
    for i in range(0,s):
        if pred[i]==tag[i] and pred[i]=='neg':
            TN = TN+1
        if pred[i]=='neg':
            m=m+1
        if pred[i]=='pos':
            n=n+1
        if pred[i]==tag[i] and pred[i]=='pos':
            TP = TP+1

    recall = float(TP)/float(n)
    print("recall: %.6f"%(recall))
    Precision = float(TP)/(float(TP)+(float(m)-float(TN)))
    print("Precision: %.6f"%(Precision))
    accuracy = (float(TP)+float(TN))/(float(m)+float(n))
    print("accuracy: %.6f"%(accuracy))
    tmp1 = float(2)*Precision*recall
    tmp2 = Precision+recall
    F1Score = tmp1/tmp2
    print("F1Score: %.6f"%(F1Score))
    TNR = float(TN)/(float(TN)+(float(m)-float(TN)))
    print("TNR: %.6f"%(TNR))
    FPR = float(1)-TNR
    print("FPR: %.6f"%(FPR))

    recall = round(recall, 6)
    Precision = round(Precision, 6)
    accuracy = round(accuracy, 6)
    F1Score = round(F1Score, 6)
    TNR = round(TNR, 6)
    FPR = round(FPR, 6)

    recall_list.append(recall)
    Precision_list.append(Precision)
    accuracy_list.append(accuracy)
    F1Score_list.append(F1Score)
    TNR_list.append(TNR)
    FPR_list.append(FPR)


def drawAUC(data, tag):
    # classifier = SklearnClassifier(SVM)
    # classifier.train(train) #训练分类器
    # pred = classifier.classify_many(data)
    clf_proba = SVC().fit(data,tag)
    FPR, recall, thresholds = roc_curve(tag,clf_proba.decision_function(data), pos_label=1)
    area = AUC(data,clf_proba.decision_function(data))
    print(area)     # 0.9696400000000001

    # 画出ROC曲线
    plt.figure()
    plt.plot(FPR, recall, color='red',label='ROC curve (area = %0.6f)' % area)
    plt.plot([0, 1], [0, 1], color='black', linestyle='--')
    # 为了让曲线不黏在图的边缘
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('Recall')
    plt.title('SVM under bag of words')
    plt.legend(loc="lower right")
    plt.show()

    # 利用ROC曲线找出最佳阀值
    maxindex = (recall - FPR).tolist().index(max(recall - FPR))
    print(thresholds[maxindex])     # -1.0860191749391461



if __name__ == "__main__":
    dataPrePro1 = dataPrePro()
    posFeatures,negFeatures = dataPrePro1.build_features()#build_feature 调换使用的分类器
    shuffle(posFeatures) #把文本的排列随机化
    shuffle(negFeatures) #把文本的排列随机化
    lenPos = len(posFeatures)
    lenNeg = len(negFeatures)
    train = posFeatures[int(lenPos*0.2):]+negFeatures[int(lenNeg*0.2):]#训练集(80%)
    test = posFeatures[:int(lenPos*0.2)]+negFeatures[:int(lenNeg*0.2)]#预测集(验证集)(20%)
    data,tag = zip(*test)#分离测试集合的数据和标签，便于验证和测试 [(data ),(label)]
    
    recall_list = []
    Precision_list = []
    accuracy_list = []
    F1Score_list = []
    TNR_list = []
    FPR_list = []

    print('BernoulliNB')
    score(BernoulliNB())
    print('MultinomiaNB')
    score(MultinomialNB())
    print('LogisticRegression')
    score(LogisticRegression())
    print('SVC')
    score(SVC())
    print('LinearSVC')
    score(LinearSVC())
    print('NuSVC')
    score(NuSVC())

    print("recall_list:")
    print(recall_list)

    print("Precision_list:")
    print(Precision_list)

    print("accuracy_list:")
    print(accuracy_list)

    print("F1Score_list:")
    print(F1Score_list)

    print("TNR_list:")
    print(TNR_list)

    print("FPR_list:")
    print(FPR_list)
    # drawAUC(data, tag)

