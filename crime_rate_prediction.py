import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures


def load_data(file_name):
    df = pd.read_csv(filepath_or_buffer=file_name)
    # Before separating data into features and labels, check for missing values.
    # missing values will be replaced by the median of that specific column.
    # Column RM has 5 missing values
    df['RM'].fillna(value=df['RM'].median(), inplace=True)

    X = df.drop('CRIM', axis=1).values
    y = df['CRIM'].values

    return X, y, df


def normalize(data):
    min_val = np.min(data[:, :], axis=0)
    max_val = np.max(data[:, :], axis=0)
    norm_x = np.array([(row - min_val) / (max_val - min_val) for row in data[:, :]])
    return norm_x


def correlation_heatmap(dataframe):
    corr_mat = dataframe[dataframe.keys()].corr()
    sns.heatmap(corr_mat, cmap='RdBu_r', robust=True, annot=True)
    plt.show()


def compute_cost(features, labels, parameters):
    m = labels.size
    h_x = np.dot(features, parameters)
    cost = (1 / (2 * m)) * sum(np.square(h_x - labels))
    return cost


def gradient_descent(features, labels, epochs, learning_rate):
    J = []
    m = labels.size
    thetas = np.random.random(size=features[0, :].shape)
    for _ in range(epochs):
        h_x = np.dot(features, thetas)
        for i in range(len(thetas)):
            thetas[i] -= (learning_rate / m) * sum((h_x - labels) * features[:, i])
        J.append(compute_cost(features=features, labels=labels, parameters=thetas))

    return thetas, J


def visualize_loss(cost):
    plt.plot(cost)
    plt.xlabel('Number of iterations')
    plt.ylabel('Cost J')
    plt.show()


def MSE(actual, prediction):
    mse = np.sum(np.square(actual - prediction))
    return mse / len(actual)


def main():
    X, y, dataFrame = load_data(file_name='./realEstate.csv')
    print(f"Data shape: {X.shape}\nLabel shape: {y.shape}")
    norm_x = normalize(data=X)
    transformer = PolynomialFeatures(degree=2)
    poly_features = transformer.fit_transform(norm_x)
    print(f"Data shape after adding poly features: {poly_features.shape}\nLabel shape after adding poly features: {y.shape}")
    # correlation_heatmap(dataframe=dataFrame)
    x_train, x_test, y_train, y_test = train_test_split(poly_features, y, test_size=0.2, shuffle=True, random_state=42)

    alpha = 0.01
    epochs = 10000
    thetas, J = gradient_descent(features=x_train, labels=y_train, epochs=epochs, learning_rate=alpha)
    visualize_loss(cost=J)
    print("MSE of training set: {}".format(MSE(y_train, np.dot(x_train, thetas))))
    print("MSE of testing set: {}".format(MSE(y_test, np.dot(x_test, thetas))))


if __name__ == '__main__':
    main()
