from django.shortcuts import render
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

def login(request):
    return render(request, r"网页设计/demonWeb.html")

def reg(request):
    if request.method=='POST':
        rank = request.POST.get('rank')
        gre = request.POST.get('gre')
        gpa = request.POST.get('gpa')
    c_prestige=int(rank)
    c_gre=int(gre)
    c_gpa=float(gpa)

    cX_test = [[c_gre, c_gpa, c_prestige]]

    df = pd.read_csv(r"C:\Users\liyun\PycharmProjects\python大数据分析\kaoyan\kaoyan\templates\binary.csv")
    df.rename(columns={"rank": "prestige"}, inplace=True)

    # 数据集按照4:1划分训练集和测试集
    X = df.ix[:, 1:]
    Y = df.admit
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=40)

    def print_cm_accuracy(Y_true, Y_pred):
        cnf_matrix = confusion_matrix(Y_true, Y_pred)
        # 输出混淆矩阵：每一行之和表示该类别的真实样本数量，
        # 每一列之和表示被预测为该类别的样本数量
        print("混淆矩阵为：")
        print(cnf_matrix, '\n')
        accuracy_percent = accuracy_score(Y_true, Y_pred)
        print("预测精确度为: %s%s" % (accuracy_percent * 100, '%'))

    knn = KNeighborsClassifier()
    knn.fit(X_train, Y_train)

    Y_pred1 = knn.predict(X_test)
    print_cm_accuracy(Y_test, Y_pred1)

    Y_pred = knn.predict(cX_test)
    print(Y_pred)

    if Y_pred == 0:
        result="别灰心，还请继续努力哦"
    else:
        result="真棒！你很有可能被录取哦"

    return render(request, r"网页设计/demonWeb.html", {"result":result})
