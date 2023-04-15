import numpy as np
import random
from typing import Union
from fastapi import FastAPI
from fastapi.responses import FileResponse
import cv2
from cv2 import imread
from starlette.responses import StreamingResponse
import io
import matplotlib
import time

import ScikitLearn
import unsupervise_test
import picture
import matrix

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/1_matrix")
def test_matrix(cols, rows):
    matrix_1 = matrix.Matrix(int(cols), int(rows))
    results={
    "La matriz A es": str([matrix_1.rows,matrix_1.columns]),
    "La traza de A es":str(matrix_1.trace()),
    "El rango de A es": str(matrix_1.rank()),
    "El determinate de A es": str(matrix_1.determinant()),
    "La inversa de A es": str(matrix_1.invert()),
    "Los valores y vectores propios de ATA son": str(matrix_1.value_vectors_eigen()[0]),
    "Los valores y vectores propios de AAT son": str(matrix_1.value_vectors_eigen()[1]),
    }
    return results

@app.get("/2_my_image")
def read_item():
    pic = picture.Pictures()
    pic.save_my_image()
    img = cv2.imread(r"C:\Users\DELL\Desktop\LabI_DimRed\resources\Results\my_image.png")
    res, image = cv2.imencode(".png", img)
    return StreamingResponse(io.BytesIO(image.tobytes()), media_type = "image/jpeg")

@app.get("/2_mean_image")
def read_averange_item():
    pic = picture.Pictures()
    pic.save_average_image()
    img = cv2.imread(r"C:\Users\DELL\Desktop\LabI_DimRed\resources\Results\average_image.png")
    res, image = cv2.imencode(".png", img)
    return StreamingResponse(io.BytesIO(image.tobytes()), media_type = "image/jpeg")

@app.get("/2_calculation_mean")
def culculation_dif():
    pic = picture.Pictures()
    pic.calculate_distance_my_picture_to_avg()
    dif = pic.calculate_distance_my_picture_to_avg()
    results={
    "La distancia entre mi foto y el promedio es": dif
    }
    return results

@app.get("/4_svd_my_image")
def svd_my_image():
    pic = picture.Pictures()
    my_image = pic.get_my_image()
    unsupervise_test.descompose_my_image(my_image)
    img = cv2.imread(r"C:\Users\DELL\Desktop\LabI_DimRed\resources\Results\my_image_svd.png")
    res, image = cv2.imencode(".png", img)
    return StreamingResponse(io.BytesIO(image.tobytes()), media_type = "image/jpeg")

@app.get("/4_diference_my_image")
def sdiference_my_image():
    pic = picture.Pictures()
    my_image = pic.get_my_image()
    response = unsupervise_test.difference_my_image(my_image)
    return response

@app.get("/5_mnist_dataset_LR")
def train_mnist_dataset_with_scikit_learn():
    X_train, y_train, X_test, y_test = ScikitLearn.load_nmist_dataset()
    start_time = time.time()
    score = ScikitLearn.train_logistic_regression_model(X_train, y_train, X_test, y_test)
    return {"Score logistic regression": score,
            "Execution time (s)": time.time() - start_time}

@app.get("/6_plot_scratch")
def plot_two_features_generated_methods_from_scratch():
    plot = unsupervise_test.plot_dimension_reduction_methods_from_scratch()
    return FileResponse(plot, media_type="image/jpg")

@app.get("/6_train-LR-unsupervised")
def train_mnist_dataset_unsupervised():
    X_train, y_train, X_test, y_test = ScikitLearn.load_nmist_dataset()

    X_train_svd = unsupervise_test.SVD_from_scratch(X_train)
    X_test_svd = unsupervise_test.SVD_from_scratch(X_test)
    start_time_model_svd = time.time()
    model_svd = ScikitLearn.train_logistic_regression_model(X_train_svd, y_train, X_test_svd, y_test)
    time_model_svd = time.time() - start_time_model_svd

    X_train_pca = unsupervise_test.PCA_from_scratch(X_train)
    X_test_pca = unsupervise_test.PCA_from_scratch(X_test)
    start_time_model_pca = time.time()
    model_pca = ScikitLearn.train_logistic_regression_model(X_train_pca, y_train, X_test_pca, y_test)
    time_model_pca = time.time() - start_time_model_pca

    X_train_tsne = unsupervise_test.TSNE_from_scratch(X_train)
    X_test_tsne = unsupervise_test.TSNE_from_scratch(X_test)
    start_time_model_tsne = time.time()
    model_tsne = ScikitLearn.train_logistic_regression_model(X_train_tsne, y_train[:100], X_test_tsne, y_test[:100])
    time_model_tsne = time.time() - start_time_model_tsne

    return {
        "Score logistic regression with SVD": model_svd,
        "Execution time fit model with SVD (s)": time_model_svd,
        "Score logistic regression with PCA": model_pca,
        "Execution time fit model with PCA (s)": time_model_pca,
        "Score logistic regression with TSNE": model_tsne,
        "Execution time fit model with TSNE (s)": time_model_tsne
    }

@app.get("/7_plot_scikit_learn")
def plot_two_features_generated_by_scikit_learn():
    plot = ScikitLearn.plot_dimension_reduction_scikit_learn()
    return FileResponse(plot, media_type="image/jpg")



def train_mnist_dataset_with_scikit_learn():
    X_train, y_train, X_test, y_test = ScikitLearn.load_nmist_dataset()

    X_train_svd = ScikitLearn.SVD_scikit_learn(X_train, X_test)
    X_test_svd = ScikitLearn.SVD_scikit_learn(X_test, X_test)
    start_time_model_svd = time.time()
    model_svd = ScikitLearn.train_logistic_regression_model(X_train_svd, y_train, X_test_svd, y_test)
    time_model_svd = time.time() - start_time_model_svd

    X_train_pca = ScikitLearn.PCA_scikit_learn(X_train)
    X_test_pca = ScikitLearn.PCA_scikit_learn(X_test)
    start_time_model_pca = time.time()
    model_pca = ScikitLearn.train_logistic_regression_model(X_train_pca, y_train)
    time_model_pca = time.time() - start_time_model_pca

    X_train_tsne = ScikitLearn.TSNE_scikit_learn(X_train)
    X_test_tsne = ScikitLearn.TSNE_scikit_learn(X_test)
    start_time_model_tsne = time.time()
    model_tsne = ScikitLearn.train_logistic_regression_model(X_train_tsne, y_train)
    time_model_tsne = time.time() - start_time_model_tsne

    return {
        "Score logistic regression with SVD": model_svd,
        "Execution time fit model with SVD (s)": time_model_svd,
        "Score logistic regression with PCA": model_pca,
        "Execution time fit model with PCA (s)": time_model_pca,
        "Score logistic regression with TSNE": model_tsne,
        "Execution time fit model with TSNE (s)": time_model_tsne
    }
#In scores they are different, but skitlearn's are a bit better Because this library is optimized, 
#updated and improved in each version.
