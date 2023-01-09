import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans

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
        axs[i].imshow(img, cmap='gray')
        axs[i].axis('off')
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


def run_example_img(name, k=2):
    assert name in IMG_NAMES + SECONDARY_IMG_NAMES
    img = imread(name)
    kmeans_segmented = run_kmeans(img, k, plot=True)
    # otsu_segmented = run_otsu(img, plot=True)
    # print(kmeans_segmented == otsu_segmented)


def run_all_imgs():
    imgs = [imread(name) for name in IMG_NAMES]
    kmeans_results = [run_kmeans(img) for img in imgs]
    otsu_results = [run_otsu(img) for img in imgs]
    imshow_row(kmeans_results, titles=IMG_NAMES, suptitle='K-means segmentation')
    imshow_row(otsu_results, titles=IMG_NAMES, suptitle='Otsu segmentation')


def main():
    # run_example_img(IMG_NAMES[0], 2)
    run_all_imgs()


if __name__ == '__main__':
    main()
