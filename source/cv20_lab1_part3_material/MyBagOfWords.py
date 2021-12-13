from sklearn.cluster import KMeans
from scipy.spatial import distance
from numpy import linalg as LA
import numpy as np

def kmeans(k, descriptor_arr):
    kmeans = KMeans(n_clusters = k)
    kmeans.fit(descriptor_arr)
    visual_words = kmeans.cluster_centers_
    return visual_words


def MyBagOfWords(data_train, data_test):
    descriptor_arr = np.vstack(data_train)
    descriptor_arr = descriptor_arr[np.random.choice(descriptor_arr.shape[0], int(np.ceil(descriptor_arr.shape[0]/2)), replace=False), :]
    num_of_centers = 500
    visual_words = kmeans(num_of_centers, descriptor_arr)
    train_hist = np.zeros([len(data_train),num_of_centers])
    for i in range(len(data_train)):
        pos = np.zeros([data_train[i].shape[0],1])
        for desc in range(len(pos)):
            distances = np.zeros([num_of_centers,1])
            for j in range(num_of_centers):
                distances[j] = distance.euclidean(data_train[i][desc,:],visual_words[j])
            pos[desc] = np.argmin(distances)
        train_hist[i,:] = np.histogram(pos,[i for i in range(num_of_centers+1)])[0]
    L = LA.norm(train_hist)
    train_hist = train_hist/L

    test_hist = np.zeros([len(data_test),num_of_centers])
    for i in range(len(data_test)):
        pos = np.zeros([data_test[i].shape[0],1])
        for desc in range(len(pos)):
            distances = np.zeros([num_of_centers,1])
            for j in range(num_of_centers):
                distances[j] = distance.euclidean(data_test[i][desc,:],visual_words[j])
            pos[desc] = np.argmin(distances)
        test_hist[i,:] = np.histogram(pos,[i for i in range(num_of_centers+1)])[0]
    L = LA.norm(test_hist)
    test_hist = test_hist/L

    return train_hist,test_hist
