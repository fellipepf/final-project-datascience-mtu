from sklearn.preprocessing import StandardScaler

'''
https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html

https://machinelearningmastery.com/standardscaler-and-minmaxscaler-transforms-in-python/
'''


def example_1():
    data = [[0, 0], [0, 0], [1, 1], [1, 1]]

    scaler = StandardScaler()
    print(scaler.fit(data))
    print(scaler.mean_)

    print(scaler.transform(data))
    print(scaler.transform([[2, 2]]))


def example_2():
    # example of a standardization
    from numpy import asarray
    from sklearn.preprocessing import StandardScaler
    # define data
    data = asarray([[100, 0.001],
                    [8, 0.05],
                    [50, 0.005],
                    [88, 0.07],
                    [4, 0.1]])
    print(data)
    # define standard scaler
    scaler = StandardScaler()
    # transform data
    scaled = scaler.fit_transform(data)
    print(scaled)


if __name__ == '__main__':
    #example_1()
    example_2()