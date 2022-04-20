import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pylab import mpl
import seaborn as sns
import data

mpl.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.figure(facecolor='snow')
sns.set_style("darkgrid", {"font.sans-serif": ['simhei', 'Droid Sans Fallback','Times New Roman']})
font_size = 12
font_family = "SimHei"
font_weight = "bold"
plt.rc("font", size=font_size, family=font_family, weight=font_weight)
#  my_font = fm.FontProperties(fname="C:\\Windows\\Fonts\\simhei.ttf")
 
font2 = {'family' : 'Times New Roman',
'weight' : 'bold',
'size'   : 12,
}

# fig,ax=plt.subplots(figsize=(6.4,4.8), dpi=100)
fig,ax=plt.subplots(figsize=(6.4,4.8), dpi=100)

def drawLineStyle():
    #设置自变量的范围和个数
    x = ["BerNB", "MultiNB", "LogReg", "SVC" ,"LSVC", "NuSVC"]
    #画图
    ax.plot(x,data.bigram_accuracy_list, label='bigram', linestyle='-', marker='*',  markersize='10')
    ax.plot(x,data.bigram_words_accuracy_list, label='bigram_words', linestyle='--', marker='p', markersize='10')
    ax.plot(x,data.jieba_feature_accuracy_list, label='jieba_feature', linestyle='-.', marker='o', markersize='10')
    ax.plot(x,data.bag_of_words_accuracy_list, label='bag_of_words', linestyle=':', marker='x', markersize='10')
    #设置坐标轴
    #ax.set_xlim(0, 9.5)
    #ax.set_ylim(0, 1.4)
    ax.set_xlabel('Classifier', font2)
    ax.set_ylabel('Accuracy', font2)
    # ax.set_ylabel('F1-score', fontsize=15)
    #设置刻度
    ax.tick_params(axis='both', labelsize=12)
    #显示网格
    #ax.grid(True, linestyle='-.')
    ax.yaxis.grid(True, linestyle='-.')
    #添加图例
    legend = ax.legend(loc='best')
    
    plt.show()
    fig.savefig('res/pic/1.png',format='png', bbox_inches='tight', transparent=False, dpi=600)

def myPie():
    data = [1249,1462,1728,1771,978]
    labels = ['2015','2016','2017','2018','2019']
    explode = [0,0,0.3,0,0]
    colors = ['red','hotpink','purple','orange','yellow']
    plt.axes(aspect="equal") #保证正圆
    plt.xlim(0,8)
    plt.ylim(0,8)

    plt.gca().spines['right'].set_color('white')
    plt.gca().spines['top'].set_color('white')
    plt.gca().spines['left'].set_color('white')
    plt.gca().spines['bottom'].set_color('white')

    # 绘制饼图
    plt.pie(x=data,  # 绘制数据
            labels=labels,  # 添加编程语言标签
            explode=explode,  # 突出显示Python
            colors=colors,  # 设置自定义填充色
            autopct='%.2f%%',  # 设置百分比的格式,保留2位小数
            pctdistance=0.8,  # 设置百分比标签和圆心的距离
            labeldistance=1.37,  # 设置标签和圆心的距离
            startangle=160,  # 设置饼图的初始角度
            center=(4, 4),  # 设置饼图的圆心(相当于X轴和Y轴的范围)
            radius=2.6,  # 设置饼图的半径(相当于X轴和Y轴的范围)
            counterclock=False,  # 是否为逆时针方向,False表示顺时针方向
            wedgeprops={'linewidth': 1, 'edgecolor': 'pink'},  # 设置饼图内外边界的属性值
            # textprops={'fontsize': 19, 'color': 'black', 'fontproperties': my_font},  # 设置文本标签的属性值
            textprops={'fontsize': 10, 'color': 'black'},  # 设置文本标签的属性值
            frame=1)  # 是否显示饼图的圆圈,1为显示

    # 不显示X轴、Y轴的刻度值
    plt.xticks(())
    plt.yticks(())
    # 添加图形标题
    # plt.title('专利分布', fontproperties=my_font)
    # plt.title('distribution of Food')
    # 显示图形
    plt.show()
    fig.savefig('res/pic/4.png')

def axPie():
    fig, ax = plt.subplots(figsize=(8, 4), subplot_kw=dict(aspect="equal"))
    recipe  = ['2015y 1249',
               '2016y 1462',
               '2017y 1728',
               '2018y 1771',
               '2019y 978']
    data = [1249,1462,1728,1771,978]

    """
    参数wedgeprops以字典形式传递，设置饼图边界的相关属性，例如圆环宽度0.5
    饼状图默认从x轴正向沿逆时针绘图，参数startangle可指定新的角（例如负40度）度起画
    """
    wedges, texts = ax.pie(data, wedgeprops=dict(width=0.5), startangle=-40)

    # 创建字典bbox_props，设置文本框的边框样式(boxstyle：方框，pad设置方框尺寸)、前景色(fc)为白色(w)、边框颜色(ec)为黑色(k)、线粗(lw)为0.72
    bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)

    """
    参数集kw以字典形式传递，包含一系列用于绘图标注的指定参数
    xycoords用于指定点xy的坐标环境，xycoords='data'表示沿用被注释的对象所采用的坐标系（默认设置）
    textcoords用于指定点xytext的坐标环境，textcoords='data'表示沿用被注释的对象所采用的坐标系（默认设置）
    参数arrowprops以字典形式传递，用于控制箭头的诸多属性，如箭头类型(arrowstyle)、箭头连接时的弯曲程度(connectionstyle)
    """
    kw = dict(xycoords='data', textcoords='data', arrowprops=dict(arrowstyle="-"),
              bbox=bbox_props, zorder=0, va="center")

    for i, p in enumerate(wedges):  # 遍历每一个扇形

        ang = (p.theta2 - p.theta1) / 2. + p.theta1  # 锁定扇形夹角的中间位置，对应的度数为ang

        # np.deg2rad(x)将度数x转为弧度(x*pi)/180
        y = np.sin(np.deg2rad(ang))  # np.sin()求正弦
        x = np.cos(np.deg2rad(ang))  # np.cos()求余弦

        """
        np.sign()符号函数：大于0返回1.0，小于0返回-1.0，等于0返回0.0
        参数horizontalalignment用于设置垂直对齐方式，可选参数：left、right、center
        当余弦值x大于0（即标签在饼图右侧时，按框左侧对齐）时，horizontalalignment="left"
        当余弦值x小于0（即标签在饼图左侧时，按框右侧对齐）时，horizontalalignment="right" 
        """
        horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]

        connectionstyle = "angle,angleA=0,angleB={}".format(ang)  # 参数connectionstyle用于控制箭头连接时的弯曲程度
        kw["arrowprops"].update({"connectionstyle": connectionstyle})  # 将connectionstyle更新至参数集kw的参数arrowprops中

        """
        用一个箭头/横线指向要注释的地方，再写上一段话的行为，叫做annotate
        ax.annotate()用于对已绘制的图形做标注
        recipe[i]是第i个注释文本
        size设置字体大小
        xy=(x1, y1)表示在给定的xycoords中，被注释对象的坐标点
        xytext=(x2, y2)表示在给定的textcoords中，注释文本的坐标点
        """
        ax.annotate(recipe[i], size=15, xy=(x, y), xytext=(1.35 * np.sign(x), 1.4 * y),
                    horizontalalignment=horizontalalignment, **kw)

    ax.set_title("")
    plt.show()
    fig.savefig('res/pic/3.png')

from pyecharts import Line, Bar
# pip install pyecharts==0.1.9.4
def barDemo():
    list1 = data.bigram_F1Score_list #bigram
    list2 = data.bigram_words_F1Score_list #bigram_words
    list3 = data.jieba_feature_F1Score_list #jieba_feature
    list4 = data.bag_of_words_F1Score_list #bag_of_words
    # labes1 = ["bigram","bigram_words","jieba_feature","bag_of_words"]
    labes1 = ["BerNB", "MultiNB", "LogReg", "SVC" ,"LSVC", "NuSVC"]
    line = Line()
    line.add("bigram",labes1,list1,is_label_show=True)
    line.add("bigram_words",labes1,list2,is_label_show=True)
    line.add("jieba_feature",labes1,list3,is_label_show=True)
    line.add("bag_of_words",labes1,list4,is_label_show=True)
    line.render()

    bar = Bar()
    bar.add("bigram",labes1,list1)
    bar.add("bigram_words",labes1,list2)
    bar.add("jieba_feature",labes1,list3)
    bar.add("bag_of_words",labes1,list4)
    bar.render("res/2.html")

def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{:.2f}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, -7),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom',rotation=45)    # 'vertical'

def barDemo_notHtml():
    x = np.arange(3,21,3) 
    width = 0.6 
    axes_width = 3  # 刻度线宽度
    axes_length = 6  # 刻度线长度
    spines_width = 3  # 坐标轴的宽度
    fig, ax = plt.subplots(figsize=(6.4,4.8), dpi=500)
    labels = ["BerNB", "MultiNB", "LogReg", "SVC" ,"LSVC", "NuSVC"]
    rects1 = ax.bar(x - width*1.5, data.bigram_F1Score_list, width, color='#FF7F0E', alpha=0.6,label='bigram')  # 画第一批数据
    rects2 = ax.bar(x - width/2, data.jieba_feature_F1Score_list, width, color='#1F77B4', alpha=1,label='jieba')  # 画第二批数据
    rects3 = ax.bar(x + width/2, data.bag_of_words_F1Score_list, width, color='r', alpha=0.6,label='bag')  # 画第一批数据
    rects4 = ax.bar(x + width *1.5 , data.bigram_words_F1Score_list, width, color='c', alpha=1,label='bigram_words')  # 画第二批数据

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('F1-Scores', fontsize="large", weight=font_weight, family = "Arial")
    # ax.set_title('Scores by group and gender', fontsize="large", weight=font_weight, family="Arial")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    # Left border is always shown
    # ax.spines["left"].set_linewidth(spines_width)
    for key in ("top", "bottom", "right","left"):
    # axes.spines[key].set_visible(False)
        ax.spines[key].set_linewidth(spines_width)

    ax.tick_params(axis="y", width=axes_width, length = axes_length)  # 刻度线
    ax.tick_params(axis="x", width=axes_width, length = axes_length)
    # plt.grid(axis='y') # 网格线 x,y,both  ,有点问题
    autolabel(rects1)   # 添加 标注
    autolabel(rects2)
    autolabel(rects3)   # 添加 标注
    autolabel(rects4)
    fig.tight_layout()

    # patches = [ mpatches.Patch(color=color[i], label="{:s}".format(labels[i]) ) for i in range(len(color)) ]
    ax=plt.gca()
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width , box.height])
    #下面一行中bbox_to_anchor指定了legend的位置
    # ax.legend(handles=patches, bbox_to_anchor=(0.95,1.12), ncol=4) #生成legend
    
    legend = ax.legend(edgecolor="w",bbox_to_anchor=(0.85,1.1), ncol=4)
    frame = legend.get_frame()
    frame.set_alpha(1)
    frame.set_facecolor('none') # 设置图例legend背景透明
    plt.show()
    # 保存为透明背景的图片
    fig.savefig('res/pic/4.png', format='png', bbox_inches='tight', transparent=True, dpi=600) 



if __name__ == "__main__":
    drawLineStyle()
    # histDemo()
    # myPie()
    # barDemo()
    # barDemo_notHtml()