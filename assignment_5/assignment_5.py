import copy

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

from collections import Counter
from sklearn.cluster import KMeans
from skimage.segmentation import chan_vese

IMG_PATH = './src/'
IMG_NAMES = ['camera', 'coins', 'rocksample', 'page']
SECONDARY_IMG_NAMES = ['Checkerboard', 'lenna']


def imread(name):
    return cv.imread(f'{IMG_PATH}{name}.png', cv.IMREAD_GRAYSCALE)


def imshow_gray(img, title=''):
    plt.imshow(img, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()


def imshow_row(imgs, titles, suptitle=''):
    length = len(imgs)
    fig, axs = plt.subplots(1, length, figsize=(16, 9))
    for i, img in enumerate(imgs):
        try:
            axs[i].imshow(img, cmap='gray')
            axs[i].axis('off')
        except TypeError:
            axs[i].plot(img)

        axs[i].set_title(titles[i])
    if suptitle:
        plt.suptitle(suptitle)
    plt.show()


def kmeans_clustering(img, k=2):
    X = img.reshape((-1, 1))
    k_m = KMeans(n_clusters=k)
    k_m.fit(X)
    values = k_m.cluster_centers_.squeeze()
    labels = k_m.labels_
    return values, labels


def run_kmeans(img, k=2, plot=False):
    values, labels = kmeans_clustering(img, k)
    img_segm = np.choose(labels, values)
    img_segm.shape = img.shape
    if plot:
        imshow_row([img, img_segm], ['Original', 'K-means segmented'])
    return img_segm


def otsu_thresholding(img):
    ret, th = cv.threshold(img, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    return ret, th


def run_otsu(img, plot=False):
    ret, th = otsu_thresholding(img)
    if plot:
        imshow_row([img, th], ['Original', 'Otsu segmented'])
    return th


def cleaning_alg(img, nn=4, iterations=1, threshold=4):
    assert threshold <= nn
    cleaned_img = np.copy(img)
    for it in range(iterations):
        for row in range(1, cleaned_img.shape[0] - 1):
            for col in range(1, cleaned_img.shape[1] - 1):
                if nn == 4:
                    neighbours = [cleaned_img[row - 1][col],
                                  cleaned_img[row][col + 1],
                                  cleaned_img[row + 1][col],
                                  cleaned_img[row][col - 1]]
                elif nn == 8:
                    neighbours = [cleaned_img[row - 1][col],
                                  cleaned_img[row][col + 1],
                                  cleaned_img[row + 1][col],
                                  cleaned_img[row][col - 1],
                                  cleaned_img[row - 1][col + 1],
                                  cleaned_img[row + 1][col + 1],
                                  cleaned_img[row + 1][col - 1],
                                  cleaned_img[row - 1][col - 1]]
                vote_count = Counter(neighbours)
                max_votes = sorted(vote_count.items(), key=lambda v: v[1])
                max_vote = max_votes[-1]
                if max_vote[1] >= threshold and max_vote[0] != cleaned_img[row][col]:
                    cleaned_img[row][col] = max_vote[0]
    return cleaned_img


def run_cleaning(img, nn=4, it=1, threshold=4):
    cleaned = cleaning_alg(img, nn, it, threshold)
    imshow_row([img, cleaned], ["Original Image", "Cleaned Image"])


def chan_vese_segmentation(img):
    return chan_vese(img, mu=0.25, lambda1=1, lambda2=1, tol=1e-3,
                     max_num_iter=200, dt=0.5, init_level_set="checkerboard",
                     extended_output=True)


def run_chan_vese(name):
    img = imread(name)
    cv = chan_vese_segmentation(img)

    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    ax = axes.flatten()

    ax[0].imshow(img, cmap="gray")
    ax[0].set_axis_off()
    ax[0].set_title("Original Image", fontsize=12)

    ax[1].imshow(cv[0], cmap="gray")
    ax[1].set_axis_off()
    title = f'Chan-Vese segmentation - {len(cv[2])} iterations'
    ax[1].set_title(title, fontsize=12)

    ax[2].imshow(cv[1], cmap="gray")
    ax[2].set_axis_off()
    ax[2].set_title("Final Level Set", fontsize=12)

    ax[3].plot(cv[2])
    ax[3].set_title("Evolution of energy over iterations", fontsize=12)

    fig.tight_layout()
    plt.show()

    return cv


def run_example_img(name, k=2):
    assert name in IMG_NAMES + SECONDARY_IMG_NAMES
    img = imread(name)
    kmeans_segmented = run_kmeans(img, k, plot=True)
    otsu_segmented = run_otsu(img, plot=True)
    return kmeans_segmented, otsu_segmented


def run_all_imgs():
    imgs = [imread(name) for name in IMG_NAMES]
    kmeans_results = [run_kmeans(img) for img in imgs]
    otsu_results = [run_otsu(img) for img in imgs]
    imshow_row(kmeans_results, titles=IMG_NAMES, suptitle='K-means segmentation')
    imshow_row(otsu_results, titles=IMG_NAMES, suptitle='Otsu segmentation')


def main():
    # run_example_img(IMG_NAMES[0], 2)
    # run_all_imgs()
    img = imread(IMG_NAMES[0])
    km = run_kmeans(img)
    run_cleaning(img=km, nn=8, it=10, threshold=4)
    # run_chan_vese(IMG_NAMES[0])


if __name__ == '__main__':
    main()
