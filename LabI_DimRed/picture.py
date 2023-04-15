import os
import cv2
import numpy as np

class Pictures:
    my_image = []
    averange_image = []

    def __init__(self):
        module_path = os.path.dirname(os.path.abspath(__file__))
        resources_path = os.path.join(module_path, 'resources\imagenes')
        self.directory = resources_path
        self.images = self.load_images()
        self.cara0 = self.images[0]
        self.my_image = []
        self.average =[]

    
    def load_images(self):
        images = []
        for filename in os.listdir(self.directory):
            if filename.endswith('.jpg') or filename.endswith('.png'):
                img_path = os.path.join(self.directory, filename)
                image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                image = cv2.resize(image, (256, 256))
                images.append(image)
        return np.array(images)
    
    def save_my_image(self):
        #img = cv2.imread(r".\resources\imagenes\CamiloLR.jpeg")
        image = cv2.imread(r".\resources\imagenes\CamiloLR.jpeg", cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (256, 256))
        Pictures.my_image = image
        png_image = cv2.imencode('.png', image)[1].tobytes()

        with open(os.path.join(r".\resources\Results", 'my_image.png'), 'wb') as f:
            f.write(png_image)

    def save_average_image(self):
        image = cv2.imread(self.directory, cv2.IMREAD_GRAYSCALE)
        average_image = np.array(np.mean(self.images, axis=(0)), dtype=np.uint8)
        Pictures.averange_image =  average_image
        png_image= cv2.imencode('.png', average_image)[1].tobytes()

        with open(os.path.join(r".\resources\Results", 'average_image.png'), 'wb') as f:
            f.write(png_image)

    def get_my_image(self):
        image = cv2.imread(r".\resources\imagenes\CamiloLR.jpeg", cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (256, 256))
        return image

    def calculate_distance_my_picture_to_avg(self):
        print(Pictures.my_image.shape)
        print(Pictures.averange_image.shape)
        return np.mean((Pictures.my_image - Pictures.averange_image) ** 2)

    #As of feature 30, it is a properly reproducible image
        
pic = Pictures()
pic.save_my_image()
pic.save_average_image()

