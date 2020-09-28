import numpy as np
import imageio
from matplotlib import pyplot as plt
import sys
import os
import random
import math
import colorsys


def distance(input1, input2):
    res = 0
    for i in range(3):
        res = res + (input1[i] - input2[i]) ** 2
    return math.sqrt(res)


def mykmeans(pixels, K):
    """
    Your goal of this assignment is implementing your own K-means.
​
    Input:
        pixels: data set. Each row contains one data point. For image
        dataset, it contains 3 columns, each column corresponding to Red,
        Green, and Blue component.
​
        K: the number of desired clusters. Too high value of K may result in
        empty cluster error. Then, you need to reduce it.
​
    Output:
        class: the class assignment of each data point in pixels. The
        assignment should be 1, 2, 3, etc. For K = 5, for example, each cell
        of class should be either 1, 2, 3, 4, or 5. The output should be a
        column vector with size(pixels, 1) elements.
​
        centroid: the location of K centroids in your result. With images,
        each centroid corresponds to the representative color of each
        cluster. The output should be a matrix with size(pixels, 1) rows and
        3 columns. The range of values should be [0, 255].
    """

    h, w, c = pixels.shape
    pixels = np.reshape(pixels, (h * w, c))
    row = h * w
    centroid = np.zeros((K, 3))
    classes = np.zeros((row, 1))
    # random centroid
    for i in range(K):
        tmp = round(random.randint(1, row))
        centroid[i,] = pixels[tmp,]

    number = 1
    prevDistance = 0
    currDistance = 0
    while currDistance < prevDistance or number == 1 or number == 2:
        prevDistance = currDistance
        centroidNumber = np.zeros((K, 1))
        sum = np.zeros((K, 3))
        # distance
        for i in range(row):
            index = 1
            distanceToCentroid = distance(centroid[0,], pixels[i,])
            for j in range(1, K):
                if distance(centroid[j,], pixels[i,]) < distanceToCentroid:
                    index = j
            classes[i] = index
            centroidNumber[index] = centroidNumber[index] + 1
            sum[index,] = sum[index,] + pixels[i,]
        # update centroid
        for i in range(K):
            if (centroidNumber[i] > 0):
                centroid[i,] = sum[i,] / centroidNumber[i]
        # update distance
        currDistance = 0
        for i in range(row):
            tmp = int(classes.item(i))
            currDistance = currDistance + distance(pixels[i,], centroid[tmp])

        number = number + 1
    # modified centroid to fit the main() required
    centers = np.zeros((row, 3))
    for i in range(row):
        tmp = int(classes.item(i))
        centers[i,] = centroid[tmp,]

    return classes, centers
    # raise NotImplementedError


def mykmedoids(pixels, K):
    """
    Your goal of this assignment is implementing your own K-medoids.
    Please refer to the instructions carefully, and we encourage you to
    consult with other resources about this algorithm on the web.
​
    Input:
        pixels: data set. Each row contains one data point. For image
        dataset, it contains 3 columns, each column corresponding to Red,
        Green, and Blue component.
​
        K: the number of desired clusters. Too high value of K may result in
        empty cluster error. Then, you need to reduce it.
​
    Output:
        class: the class assignment of each data point in pixels. The
        assignment should be 1, 2, 3, etc. For K = 5, for example, each cell
        of class should be either 1, 2, 3, 4, or 5. The output should be a
        column vector with size(pixels, 1) elements.
​
        centroid: the location of K centroids in your result. With images,
        each centroid corresponds to the representative color of each
        cluster. The output should be a matrix with size(pixels, 1) rows and
        3 columns. The range of values should be [0, 255].
    """
    h, w, c = pixels.shape
    pixels = np.reshape(pixels, (h * w, c))
    row = h * w
    classes = np.zeros((row, 1))
    hsv_pixels = np.zeros((row, 3))
    hsv_centroid = np.zeros((K, 3))
    # transform to hsv
    for i in range(row):
        hsv_pixels[i,] = colorsys.rgb_to_hsv(pixels[i][0], pixels[i][1], pixels[i][2])

    # random centroid
    for i in range(K):
        tmp = round(random.randint(1, row))
        hsv_centroid[i,] = hsv_pixels[tmp,]

    number = 1
    prevDistance = 0
    currDistance = 0
    while currDistance < prevDistance or number == 1 or number == 2:
        prevDistance = currDistance
        centroidNumber = np.zeros((K, 1))
        sum = np.zeros((K, 3))
        #distance
        for i in range(row):
            index = 1
            distanceToCentroid = distance(hsv_centroid[0,], hsv_pixels[i,])
            for j in range(1, K):
                if distance(hsv_centroid[j,], hsv_pixels[i,]) < distanceToCentroid:
                    index = j
            classes[i] = index
            centroidNumber[index] = centroidNumber[index] + 1
            sum[index,] = sum[index,] + hsv_pixels[i,]
        #average
        for i in range(K):
            if centroidNumber[i] != 0:
                hsv_centroid[i,] = sum[i,] / centroidNumber[i]
        #update distance
        for i in range(K):
            distanceToCentroid = distance(hsv_centroid[i,], hsv_pixels[0,])
            index = 1
            minDistance = distanceToCentroid
            for j in range(1, row):
                distanceToCentroid = distance(hsv_centroid[i,], hsv_pixels[j,])
                if distanceToCentroid < minDistance:
                    minDistance = distanceToCentroid
                    index = j
            hsv_centroid[i,] = hsv_pixels[index,]
        #update distance
        for i in range(row):
            tmp = int(classes.item(i))
            currDistance = currDistance + distance(hsv_pixels[i,], hsv_centroid[tmp,])

        number = number + 1
    #transform back to rgb
    centroid = np.zeros((K, 3))
    for i in range(K):
        centroid[i,] = colorsys.hsv_to_rgb(hsv_centroid[i][0], hsv_centroid[i][1], hsv_centroid[i][2])

    # modified centroid to fit the main() required
    centers = np.zeros((row, 3))
    for i in range(row):
        tmp = int(classes.item(i))
        centers[i,] = centroid[tmp,]
    return classes, centers
    # raise NotImplementedError


def main():
    if (len(sys.argv) < 2):
        print("Please supply an image file")
        return

    image_file_name = sys.argv[1]
    K = 5 if len(sys.argv) == 2 else int(sys.argv[2])
    print(image_file_name, K)
    im = np.asarray(imageio.imread(image_file_name))

    fig, axs = plt.subplots(1, 2)

    classes, centers = mykmedoids(im, K)
    print(classes, centers)
    new_im = np.asarray(centers.reshape(im.shape), im.dtype)
    imageio.imwrite(os.path.basename(os.path.splitext(image_file_name)[0]) + '_converted_mykmedoids_' + str(K) +
                os.path.splitext(image_file_name)[1], new_im)
    axs[0].imshow(new_im)
    axs[0].set_title('K-medoids')

    classes, centers = mykmeans(im, K)
    print(classes, centers)
    new_im = np.asarray(centers.reshape(im.shape), im.dtype)
    imageio.imwrite(os.path.basename(os.path.splitext(image_file_name)[0]) + '_converted_mykmeans_' + str(K) +
                os.path.splitext(image_file_name)[1], new_im)
    axs[1].imshow(new_im)
    axs[1].set_title('K-means')

    plt.show()

if __name__ == '__main__':
    main()
