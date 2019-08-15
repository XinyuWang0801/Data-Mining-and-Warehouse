import helper
import sklearn


def fool_classifier(test_data):  ## Please do not change the function defination...
    ## Read the test data file, i.e., 'test_data.txt' from Present Working Directory...

    ## You are supposed to use pre-defined class: 'strategy()' in the file `helper.py` for model training (if any),
    #  and modifications limit checking
    strategy_instance = helper.strategy()

    x_train = strategy_instance.class0 + strategy_instance.class1

    import numpy as np
    y = np.zeros((540, 1), dtype=np.int)

    y[360:] = 1

    y = y.ravel()

    # 测试
    count_y1 = 0
    for x in y:
        if x == 1:
            count_y1 += 1

    # print('count_y1:',count_y1)
    # 完毕

    def createVocabList(dataSet):
        vocabSet = set([])
        for document in dataSet:
            vocabSet = vocabSet | set(document)
        return list(vocabSet)

    def setOfWords2Vec(vocabSet, inputSet):
        returnVec = [0] * len(vocabSet)
        for word in inputSet:
            if word in vocabSet:
                returnVec[vocabSet.index(word)] = 1
        return returnVec

    data = createVocabList(x_train)

    trainAll = []
    for postinDoc in x_train:
        trainAll.append(setOfWords2Vec(data, postinDoc))

    #print('trainAll_before',trainAll)
    trainAll = np.array(trainAll)
    #print('trainAll',trainAll)

    with open('test_data.txt', 'r') as test1:
        test1 = [line.strip().split(' ') for line in test1]

    testAll = []
    for postinDoc in test1:
        testAll.append(setOfWords2Vec(data, postinDoc))
    #print('testAll',testAll)

    parameters = {}
    parameters['C'] = 0.051
    parameters['kernel'] = 'linear'
    parameters['degree'] = 3
    parameters['gamma'] = 'auto'
    parameters['coef0'] = 1
    clf = strategy_instance.train_svm(parameters, trainAll, y)
    # 测试 test分类
    predict_test = clf.predict(testAll)
    print('type of testAll',testAll)
    print('predict_test:\n', predict_test)
    # print('len(predict_test):\n',len(predict_test))
    count_1 = 0
    count_0 = 0
    for x in predict_test:
        if x == 1:
             count_1 += 1
        if x == 0:
             count_0 += 1
    print('count_1: ', count_1)
    print('count_0: ', count_0)
    #   完毕
    # 测试 class0 分类
    c0 = strategy_instance.class0
    print('class0:',c0)

    c0All = []
    for postinDoc in c0:
        # print('postinDoc:',postinDoc)
        c0All.append(setOfWords2Vec(data, postinDoc))  # 对 x_train的每个样例  返回该条样例中词出现在字典list中的位置  将位置的记录list加入trainALL
    # print('c0All:',len(c0All))
    predict_c0 = clf.predict(c0All)
    print('predict_c0:\n', predict_c0)
    print('len(predict_c0):\n', len(predict_c0))
    count_c01 = 0
    for x in predict_c0:
        if x == 1:
            count_c01 += 1
    print('count_c01:', count_c01)
    #   完毕
    # 测试 class1 分类
    c1 = strategy_instance.class1
    # print('class1:',class1)

    c1All = []
    for postinDoc in c1:
        # print('postinDoc:',postinDoc)
        c1All.append(setOfWords2Vec(data, postinDoc))  # 对 x_train的每个样例  返回该条样例中词出现在字典list中的位置  将位置的记录list加入trainALL
    # print('c0All:',len(c1All))
    predict_c1 = clf.predict(c1All)
    print('predict_c1:\n', predict_c1)
    print('len(predict_c1):\n', len(predict_c1))
    count_c11 = 0
    for x in predict_c1:
        if x == 1:
            count_c11 += 1
    print('count_c11:', count_c11)
    #   完毕
    w = clf.coef_
    index = np.where(w[0] < 0)[0]

    dic_w = {}
    for i in index:
        dic_w[i] = w[0][i]
    dic_w = sorted(dic_w.items(), key=lambda d: d[1])[0:200]

    index = [dic_w[i][0] for i in range(len(dic_w))]

    add_word = []
    for i in index:
        add_word.append(data[i])

    n = 0
    for i in range(len(test1)):
        n = 0
        for w in add_word:
            if n == 20:
                break

            if w not in test1[i]:
                test1[i].append(w)

            else:
                continue
            n = n + 1

    file = open('./modified_data.txt', 'w')
    for i in range(len(test1)):
        file.write(" ".join(test1[i]))
        file.write("\n")
    file.close()
    # 测试modify
    with open('./modified_data.txt', 'r') as modified1:
        modified1 = [line.strip().split(' ') for line in modified1]

    modifiedAll = []
    for postinDoc in modified1:
        modifiedAll.append(setOfWords2Vec(data, postinDoc))  # 对test中每个样例处理 将样例中的词通过字典list位置映射起来
    predict_modified = clf.predict(modifiedAll)
    # print('predict_modified:\n',predict_modified)
    # print('len(predict_modified):\n',len(predict_modified))

    count_m0 = 0
    for x in predict_modified:
        if x == 0:
            count_m0 += 1
    # print('count_m0:',count_m0)
    # 完毕
    ## You can check that the modified text is within the modification limits.
    modified_data = './modified_data.txt'
    assert strategy_instance.check_data(test_data, modified_data)
    return strategy_instance  ## NOTE: You are required to return the instance of this class.

if __name__ == "__main__":
    test_data = './test_data.txt'
    print("start")
    strategy_instance = fool_classifier(test_data)


    ########
    #
    # Testing Script.......
    #
    #
    ########
