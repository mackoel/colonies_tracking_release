import cv2
import imutils
import numpy as np
from PIL import Image, ImageFilter
from matplotlib import pyplot as plt
from scipy import interpolate
from scipy.signal import correlate2d
import imutils
from scipy.fftpack import fftshift, ifftshift, fft2, ifft2


#
# filename_0 = "scans/96.tifclass_1.tif"
# filename_1 = "scans/97.tifclass_1.tif"


def erode(cycles, image):
    for _ in range(cycles):
        image = image.filter(ImageFilter.MinFilter(3))
    return image


def dilate(cycles, image):
    for _ in range(cycles):
        image = image.filter(ImageFilter.MaxFilter(5))
    return image


def imageOpening(path):
    f = open(path, "rb")
    chunk = f.read()
    chunk_arr = np.frombuffer(chunk, dtype=np.uint8)
    img = cv2.imdecode(chunk_arr, cv2.IMREAD_COLOR)
    # if img is not None:
    #     if img.shape[2] == 3:
    #         #print("Изображение в формате RGB")
    #     elif img.shape[2] == 4:
    #         print("Изображение в формате RGBA")
    #     else:
    #         print("Изображение не в формате RGB")
    return img


def resizingImage(cv2_image, scale_percent):
    width = int(cv2_image.shape[1] * scale_percent / 100)
    height = int(cv2_image.shape[0] * scale_percent / 100)
    dim = (width, height)

    # resize image
    resized = cv2.resize(cv2_image, dim, interpolation=cv2.INTER_AREA)

    # print('Default Dimensions : ', cv2_image.shape[1], cv2_image.shape[0])
    # print('Resized Dimensions : ', resized.shape)

    return resized


def processing(image):
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # cv2.imshow("Grayscale image", resizingImage(grayscale_image, 40))
    # cv2.waitKey(0)

    equ_image = cv2.equalizeHist(grayscale_image)
    # cv2.imshow("Equalized image", resizingImage(equ_image, 40))
    # cv2.waitKey(0)

    # ret, thresh = cv2.threshold(equ_image, 127, 255, cv2.THRESH_BINARY_INV)
    ret, thresh = cv2.threshold(equ_image, 100, 255, cv2.THRESH_BINARY)
    #INV добавлять

    image_pil = Image.fromarray(cv2.cvtColor(thresh, cv2.COLOR_BGR2RGB))
    # image_pil.show(title='PIL image')


    #erosion
    image_pil_eroded = erode(3, image_pil)
    # image_pil_eroded.show(title='PIL eroded_image')

    #blur
    image_pil_eroded_blured = cv2.GaussianBlur(cv2.cvtColor(np.array(image_pil_eroded), cv2.COLOR_RGB2BGR), (77, 77), 0)
    # cv2.imshow('image_pil_eroded_blured', resizingImage(image_pil_eroded_blured, 40))

    #treshold
    blured_treshed_image = \
        cv2.threshold(cv2.cvtColor(image_pil_eroded_blured, cv2.COLOR_RGB2GRAY), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    # cv2.imshow('BluredThreshed', resizingImage(blured_treshed_image, 40))
    # cv2.waitKey(0)

    return blured_treshed_image