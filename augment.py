import numpy as np
# from matplotlib.pyplot import imshow, hist, show,figure, subplot, subplots_adjust, setp
# from tensorflow.python.keras._impl.keras.datasets.cifar10 import load_data


def random_crop(image, crop_h, crop_w):
    h, w, _ = image.shape
    top = np.random.randint(0, h - crop_h)
    left = np.random.randint(0, w - crop_w)
    bottom = top + crop_h
    right = left + crop_w
    image = image[top:bottom, left:right, :]
    return image

def random_flip(image):
    if np.random.rand() < 0.5:
        image = image[:, ::-1, :]
    return image


def augment(input_img):
    h = input_img.shape[0]
    w = input_img.shape[1]

    input_img = np.pad(input_img, ((4, 4), (4, 4), (0, 0)), 'constant')
    distorted_img = random_crop(input_img, h, w)
    distorted_img = random_flip(distorted_img)

    return distorted_img

# (x_train, y_train), (x_test, y_test) = load_data()
#
# a=x_train[0]
#
# c=augment(a)
#
# imshow(c)
# show()