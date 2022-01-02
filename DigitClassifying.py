import numpy
from pyparsing.core import Dict
from operator import ne
from numpy.core.fromnumeric import argmin
#
# This is a sample Notebook to demonstrate how to read "MNIST Dataset"
#
import numpy as np  # linear algebra
import matplotlib.pyplot as plt

def PCA2(pics,p):
    cov = pics@pics.transpose()
    m = len(pics[0])
    cov = (1/m)*cov

    w, v = np.linalg.eig(cov)
    eigenValues , v = w.real , v.real

    x = np.arange(0, len(eigenValues))
    y = np.array(eigenValues)
    # plotting to check if decreasing
    plt.title("eigen values")
    plt.xlabel("X axis")
    plt.ylabel("Y axis")
    plt.plot(x, y, color="green")
    plt.show()

    Up = v[:,0:p]
    Up = Up.transpose()

    # new_images=[]


    new_images = numpy.array([Up@x for x in pics.transpose()])

    return new_images,Up.transpose()

def plot_pics(img , index, Up, new_images, y_train):
    img = img.reshape(28, 28)

    plt.imshow(img, cmap='gray')
    plt.title(y_train[index])
    plt.show()

    rec_img = new_images[index]
    rec_img = Up @ rec_img
    rec_img = rec_img.reshape(28, 28)
    plt.imshow(rec_img, cmap='gray')
    plt.title(y_train[index])
    plt.show()



###########----C----#######

def Kmeans(k, images, centers):
    # intialize k empty clusters
    clusters = np.empty([k],dtype=object)
    changed = True
    images_centers =np.zeros(len(images)) -2
    # run until clusters doesnt change
    # index = 0
    while (changed):
        # print(index)
        # index+=1
        changed=False
        old_clusters = clusters
        new_clusters = np.empty([k],dtype=object)
        for i in range(k):
            new_clusters[i] = list()
        for i in range(len(images)):
            distances = np.zeros(k)
            for j in range(k):
                # find distance between the image to center
                distances[j] = ((images[i]-centers[j])**2).sum()
            min_index = argmin(distances)
            # add the image to the cluster that coresponds to closest center
            new_clusters[min_index].append(images[i])
            if min_index!=images_centers[i]:
                changed = True
                images_centers[i] =min_index
        # update the centers to the mean of every cluster
        for i in range(k):
            if len(new_clusters[i]) > 0:
                sum = 0
                for j in range(len(new_clusters[i])):

                    sum = sum + new_clusters[i][j]
                mean = sum / len(new_clusters[i])
                centers[i] = mean

        # if np.array_equal(old_clusters, new_clusters): changed = False


    return clusters, centers, images_centers

def cluster_label(images_centers, labels):
    cluster_by_digits = np.empty([10],dtype=object)
    for i in range(10):
        cluster_by_digits[i] = list()
    for i in range(len(images_centers)):
        real_val = int(labels[i])
        cluster_by_digits[int(images_centers[i])].append(real_val)
    clusters_val = np.zeros(10)
    for i in range(10):
        most_common = find_most_common(cluster_by_digits[i])
        clusters_val[i] = most_common
    return clusters_val,cluster_by_digits

def find_most_common(list):
    counter = np.zeros(10)
    for x in list:
        counter[x]+=1
    return np.argmax(counter)


def success_tester(cluster_by_digits, clusters_val):
    successes = 0
    counter=0
    for i in range(10):
        for x in (cluster_by_digits[i]):
            counter+=1
            if x == clusters_val[i]:
                successes += 1
    percentage = (successes / counter) * 100
    return percentage

