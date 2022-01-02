import matplotlib.pyplot as plt
import numpy as np

import MnistDataloader
import DigitClassifying

loader = MnistDataloader.MnistDataloader('train-images.idx3-ubyte', 'train-labels.idx1-ubyte'
                                         , 't10k-images.idx3-ubyte', 't10k-labels.idx1-ubyte')
(x_train, y_train), (x_test, y_test) = loader.load_data()

# divide the images by 250 and decrease by 0.5
x_train = np.array(x_train)
x_train = (x_train / 255) - 0.5
x_test = np.array(x_test)
x_test = (x_test / 255) - 0.5


# flatten the images into vectors
x_train = np.array([x.flatten() for x in x_train]).transpose()
x_test = np.array([x.flatten() for x in x_test]).transpose()
# x_train = x_train[:,:1000]


new_images,Up = DigitClassifying.PCA2(x_train, 20)


# plot an example of an image and a reconstructed image
index = 0
img = x_train[:,index]
DigitClassifying.plot_pics(img , index, Up, new_images, y_train)

#run kmean with p=20 and randomize centers
# centers = np.random.random((10, 20)) - 0.5
# clusters,centers,changes = DigitClassifying.Kmeans(10, new_images, centers)

#test the success using test images
new_images,Up = DigitClassifying.PCA2(x_test, 20)
centers = np.random.random((10, 20)) - 0.5
clusters,centers,changes = DigitClassifying.Kmeans(10, new_images, centers)

#assign each cluster to a digit
clusters_labels,clusters_digits = DigitClassifying.cluster_label(changes, y_test)
success_rate = DigitClassifying.success_tester(clusters_digits, clusters_labels)
print("test run success rate is:")
print(success_rate)

#try the whole process 3 times
new_images,Up = DigitClassifying.PCA2(x_train, 20)
for i in range(3):
    centers = np.random.random((10, 20)) - 0.5
    clusters, centers, changes = DigitClassifying.Kmeans(10, new_images, centers)
    clusters_labels, clusters_digits = DigitClassifying.cluster_label(changes, y_train)
    success_rate = DigitClassifying.success_tester(clusters_digits, clusters_labels)
    print(i,"run success rate is:")
    print(success_rate)

#run kmean with smaller p=12 and compare result

new_images,Up = DigitClassifying.PCA2(x_train, 12)
centers = np.random.random((10, 12)) - 0.5
clusters,centers,changes = DigitClassifying.Kmeans(10, new_images, centers)
clusters_labels, clusters_digits = DigitClassifying.cluster_label(changes, y_train)
success_rate = DigitClassifying.success_tester(clusters_digits, clusters_labels)
print("p=12 success rate is:")
print(success_rate)

#run it again but initialize the centers with the mean of 10 images per label
new_images,Up = DigitClassifying.PCA2(x_train, 20)
centers = np.zeros((10,20))
for center_index in range(10):
    images_found_per_label=0
    k=0
    while images_found_per_label<10:
        if y_train[k]==center_index:
            centers[center_index]+=new_images[k]
            images_found_per_label+=1
        k+=1
    centers[center_index] = centers[center_index]/10

clusters,centers,changes = DigitClassifying.Kmeans(10, new_images, centers)
clusters_labels, clusters_digits = DigitClassifying.cluster_label(changes, y_train)
success_rate = DigitClassifying.success_tester(clusters_digits, clusters_labels)
print("centers with the mean of 10 images per label run success rate is:")
print(success_rate)