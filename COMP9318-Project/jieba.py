import helper
import numpy as np
import sklearn
from sklearn.feature_extraction.text import CountVectorizer


def fool_classifier(test_data):  ## Please do not change the function defination...
    ## Read the test data file, i.e., 'test_data.txt' from Present Working Directory...

    ## You are supposed to use pre-defined class: 'strategy()' in the file `helper.py` for model training (if any),
    #  and modifications limit checking
    vectorizer = CountVectorizer()
    strategy_instance = helper.strategy()
    x_train = []
    x_train_ = []
    list_x_train = strategy_instance.class0 + strategy_instance.class1
    for i in list_x_train:
        print(i)
        x_train_ += i

    print('x_train_: \n')
    print(x_train_)



    X = vectorizer.fit_transform(x_train)
    print(X.toarray())
    print(vectorizer.get_feature_names())


    y = np.zeros((540, 1), dtype=np.int)
    y[360:] = 1
    y = y.ravel()  # Convert a multidimensional array to a one-dimensional array
    #print(y)
    print('end of printing y')


if __name__ == "__main__":
    test_data = './test_data.txt'
    print("1")
    strategy_instance = fool_classifier(test_data)
    print('start showing results')
    print(strategy_instance)

    ########
    #
    # Testing Script.......
    #
    #
    ########

    print("hey")