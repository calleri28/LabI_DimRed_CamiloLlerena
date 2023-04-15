import time
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
from cv2 import imread

from ScikitLearn import load_nmist_dataset
from Unsupervised.SVD_Unsuper import SvdUnsupervised
from Unsupervised.PCA_Unsuper import PcaUnsupervised
from Unsupervised.Tsne_Unsuper import TsneUnsupervised

svd = SvdUnsupervised(n_components=2)

def draw_svd(u_matrix, sigma_matrix, v_matrix):
    num_sv = [1, 10, 20, 50, 140, 180, 256]

    fig, ax = plt.subplots(figsize=(9, 6))
    for i, n in enumerate(num_sv):
        # Reconstruct the image using the first n singular values
        imge = np.matmul(u_matrix[:, :n], sigma_matrix[:n, :n])
        imge = np.matmul(imge, v_matrix[:n, :])

        # Plot the reconstructed image
        plt.subplot(3, 3, i + 1)
        plt.imshow(imge, cmap='gray')
        plt.title('Singular values: {}'.format(n))
        plt.axis('off')

    plt.tight_layout()
    png_image = cv2.imencode('.png', imge)[1].tobytes()
    plt.savefig(os.path.join(r".\resources\Results", 'my_image_svd.png'))


def descompose_my_image(matrix):
    svd = SvdUnsupervised(n_components=2)
    u_matrix, sigma_matrix, v_matrix = svd.fit_svd(matrix)
    return draw_svd(u_matrix, sigma_matrix, v_matrix)

def difference_my_image(matrix):
    svd = SvdUnsupervised(n_components=2)
    u_matrix, sigma_matrix, v_matrix = svd.fit_svd(matrix)
    n = 50
    img_reconstructed = np.matmul(u_matrix[:, :n], sigma_matrix[:n, :n])
    img_reconstructed = np.matmul(img_reconstructed, v_matrix[:n, :])

    result = {"MSE - my image and the aproximation " + str(n) + " singular values:":
                        np.round(np.mean((matrix - img_reconstructed) ** 2), 3)}
    return result

#The resulting (Euclidean) distance is a measure of the similarity of the image with respect to the mean 
# image of the image set. 
#*MSE my - pro

def PCA_from_scratch(X):
    start_time = time.time()
    pca_from_scratch = PcaUnsupervised(n_components=2)
    X_pca = pca_from_scratch.fit_transform(X)
    print(f"Execution time PCA from scratch (s): {time.time() - start_time}")
    return X_pca

N = 100

def SVD_from_scratch(X):
    start_time = time.time()
    X_svd = svd.fit_transform(X)
    print(f"Execution time SVD from scratch (s): {time.time() - start_time}")
    return X_svd

def TSNE_from_scratch(X):
    start_time = time.time()
    tsne_from_scratch = TsneUnsupervised()
    X = X[:N, :]
    X_tsne = tsne_from_scratch.fit(X)
    print(f"Execution time TSNE from scratch (s): {time.time() - start_time}")
    return X_tsne

def TSNE_transform_from_scratch(X):
    start_time = time.time()
    tsne_from_scratch = TsneUnsupervised()
    X = X[:N, :]
    X_tsne = tsne_from_scratch.fit_transform(X)
    print(f"Execution time TSNE from scratch (s): {time.time() - start_time}")
    return X_tsne



def plot_dimension_reduction_methods_from_scratch():
    picture_name = "DimensionReductionFromScratch.jpg"
    path = os.path.dirname(os.path.abspath(__file__))
    resources_path = os.path.join(r".\resources", 'Results')

    X_train, y_train, X_test, y_test = load_nmist_dataset()

    train_color = ['c' if i == '0' else 'm' for i in y_train]
    test_color = ['c' if i == '0' else 'm' for i in y_test]

    svd_train_components = SVD_from_scratch(X_train)
    svd_test_components = SVD_from_scratch(X_test)
    pca_train_components = PCA_from_scratch(X_train)
    pca_test_components = PCA_from_scratch(X_test)
    tsne_train_components = TSNE_from_scratch(X_train)
    tsne_test_components = TSNE_from_scratch(X_test)

    _, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(11, 11))
    ax1.scatter(pca_train_components[:, 0], pca_train_components[:, 1], c=train_color)
    ax1.set_title('PCA train components')
    ax2.scatter(pca_test_components[:, 0], pca_test_components[:, 1], c=test_color)
    ax2.set_title('PCA test components')
    ax3.scatter(tsne_train_components[:, 0], tsne_train_components[:, 1], c=train_color[:N])
    ax3.set_title('TSNE train components')
    ax4.scatter(tsne_test_components[:, 0], tsne_test_components[:, 1], c=test_color[:N])
    ax4.set_title('TSNE test components')
    ax5.scatter(svd_train_components[:, 0], svd_train_components[:, 1], c=train_color)
    ax5.set_title('SVD train components')
    ax6.scatter(svd_test_components[:, 0], svd_test_components[:, 1], c=test_color)
    ax6.set_title('SVD test components')

    plt.tight_layout()
    plt.savefig(os.path.join(resources_path, picture_name))
    return os.path.join(resources_path, picture_name)

