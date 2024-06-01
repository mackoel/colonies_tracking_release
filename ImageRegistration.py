# import the necessary packages
import math
from multiprocessing import Pool

import numpy
import numpy as np
import cv2
# from matplotlib import pyplot as plt, image as mpimg
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import ImageProcessing as IP
from scipy.ndimage import shift
import os
import time
from joblib import Parallel, delayed
from functools import reduce
import multiprocessing
from scipy.fft import fft2, ifft2, fftshift
import CoreTrackingAPI as CT



# so far, I have left the commented-out image outputs at different stages of the script, so that it would be easier to debug later.
def circleSearch(image_circles):
    image_output = np.invert(image_circles)  # ?????

    # cv2.imshow("The result", IP.resizingImage(cv2.bitwise_not(image_output), 40))
    # cv2.waitKey(0)

    # конвертация в grayscale
    image_gray = image_output
    # cv2.imshow("The result", IP.resizingImage(image_gray, 40))
    # cv2.waitKey(0)
    image_gray_blur = cv2.medianBlur(image_gray, 5)
    # cv2.imshow("The result", IP.resizingImage(image_gray_blur, 40))
    # cv2.waitKey(0)
    # детекция кругов
    circles = cv2.HoughCircles(image_gray_blur, cv2.HOUGH_GRADIENT, 1.5, 20, param1=1, param2=20, minRadius=9,
                               maxRadius=30)
    # print(circles)
    if circles is not None:
        # конвертация координат центра и радиуса в int
        circles = np.round(circles[0, :]).astype("int")

        for (x, y, r) in circles:
            # printing data of the circle
            # print(f"Center: {x},{y}   Radius: {r}")

            # обведём найденный круг
            cv2.circle(image_output, (x, y), r, (0, 255, 0), 1)
            # и ткнём точку в центр
            cv2.circle(image_output, (x, y), 2, (255, 0, 255), -1)
        #
        # cv2.imshow("The result", IP.resizingImage(image_output, 40))
        # cv2.waitKey(0)

        # print(circles, type(circles))

    return circles


def shiftSearch(temp_circles, orig_circles):
    cnt_of_circles = 1
    shift_val = temp_circles - orig_circles
    # print(temp_circles, '-', orig_circles, '=', shift_val)
    # if i need to do boundary for shift

    # bound = 10
    # for i in range(2):
    #     for j in range(2):
    #         if abs(shift_val[i, j]) <= bound:
    #             shift_val[i, j] = 0
    return shift_val[0]


# In case I need to make a rotation
def rotation(image, angleInDegrees):
    angleInDegrees = 15
    h, w = image.shape[:2]
    print('h, w: ', h, w)
    # img_c = (145, 799)
    img_c = (w / 2, h / 2)

    rot = cv2.getRotationMatrix2D(img_c, angleInDegrees, 1)

    rad = math.radians(angleInDegrees)
    sin = math.sin(rad)
    cos = math.cos(rad)
    b_w = int((h * abs(sin)) + (w * abs(cos)))
    b_h = int((h * abs(cos)) + (w * abs(sin)))

    rot[0, 2] += ((b_w / 2) - img_c[0])
    rot[1, 2] += ((b_h / 2) - img_c[1])

    outImg = cv2.warpAffine(image, rot, (b_w, b_h), flags=cv2.INTER_LINEAR)
    return outImg


# I will need to make the correction and rendering function optional. Now it just shifts the image along the x-axis by -140
def imageCorrection(orig_image, image_shift):
    # corrected_image = shift(orig_image, image_shift, mode='constant')
    num_rows, num_cols = orig_image.shape[:2]
    # Creating a translation matrix
    translation_matrix = np.float32([[1, 0, -140], [0, 1, 0]])

    # Image translation
    img_translation = cv2.warpAffine(orig_image, translation_matrix, (num_cols, num_rows))
    return img_translation


def pairCorrection(temp_image, orig_image, base_image):
    # boundaries for cropping
    x, y = 100, 90
    dx, dy = 300, 200

    # special areas founding
    circle_temp = circleSearch(temp_image[y:y + dy, 0:x + dx])
    circle_orig = circleSearch(orig_image[y:y + dy, 0:x + dx])
    # cv2.imshow('Rotated', temp_image[y:y + dy, 0:x + dx])
    # cv2.imshow('Rotated', orig_image[y:y + dy, 0:x + dx])

    if circle_orig is None:
        # print(circle_orig, type(circle_orig))
        circle_orig = circle_temp

    if len(circle_orig) != 1:
        print("Circle length = ", len(circle_orig))
        print(
            "ERROR with detection!!!ERROR!!!ERROR!!!ERROR!!!ERROR!!!ERROR!!!ERROR!!!ERROR!!!ERROR!!!ERROR!!!ERROR!!!ERROR!!!ERROR!!!ERROR!!!ERROR!!!ERROR!!!ERROR!!!ERROR!!!ERROR!!!ERROR!!!ERROR!!!ERROR!!!ERROR!!!ERROR!!!ERROR!!!ERROR!!!ERROR!!!ERROR!!!ERROR!!!ERROR!!!ERROR!!!ERROR!!!ERROR!!!ERROR!!!ERROR!!!ERROR!!!")
        return [], []

    # will be useful for debuging
    # print("template circles: ", circle_temp)
    # print("original circles: ", circle_orig, type(circle_orig))
    # print("template circles: ", circle_temp, circle_temp[0][1])
    # print("original circles: ", circle_orig, circle_orig[0][1])

    # rotating shift
    # angle = 45
    # out_image = rotation(base_image, 45)
    # cv2.imshow('Rotated', IP.resizingImage(out_image, 40))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # default shift
    shift_val = shiftSearch(circle_temp[:, 0:2], circle_orig[:, 0:2])
    corrected_image = imageCorrection(cv2.cvtColor(np.array(base_image), cv2.COLOR_RGB2BGR), shift_val)
    # print('shift val bebebeb', shift_val)
    return corrected_image, shift_val


def SPOTdirectCorrection(directory, tag_sym=(1, 2)):
    shift_vals = list()
    file_idxes = list()
    files = [(os.path.normpath(directory + '/' + f), int(f[tag_sym[0]:tag_sym[1]])) \
             for f in os.listdir(directory) if f.endswith(".tif")]
    sorted_files = sorted(files, key=lambda t: t[1])
    file_idxes = [int(file_name[1]) for file_name in sorted_files]
    im_idx = 0
    # print(os.listdir(directory), os.listdir(directory)[0], len(os.listdir(directory)))
    time_vals = list()
    for file_name in os.listdir(directory):
        if file_name.endswith(".tif"):
            # print(file_name.endswith(".tif"))
            # Prints only text file present in My Folder
            # print(file_name)
            start_time = time.time()
            if im_idx == 0:
                original = IP.imageOpening(os.path.normpath(directory + '/' + file_name))
                # os.path.join
            temp_image = original
            original = IP.imageOpening(os.path.normpath(directory + '/' + file_name))
            template_mod = IP.processing(temp_image)
            original_mod = IP.processing(original)
            # тут иногда ошибка, изображение получается инверитированным возможно потому что серые должны быть на входе,\
            # а я 2904 запускал на черно-белых
            # cv2.imshow('temp', template_mod)
            # cv2.imshow('orig', original_mod)
            # cv2.waitKey(0)
            corrected_image, shift_val = pairCorrection(template_mod, original_mod,
                                                        cv2.cvtColor(np.array(original), cv2.COLOR_RGB2BGR))
            if (len(corrected_image) == 0) and (len(shift_val) == 0):
                print("ERROR on ", file_name, '!!!')
                print("Current shift values:")
                return shift_vals, time_vals
            # print("shift between current and first image:", shift_val)
            end_time = time.time()
            execution_time = end_time - start_time
            time_vals.append(execution_time)
            shift_vals.append(shift_val.tolist())
            im_idx += 1
    print('SHIFT_VALS:', shift_vals)
    print('TIME_VALS:', time_vals, sum(time_vals), sep='\n')
    return shift_vals, time_vals, file_idxes


def writingShifts(saving_directory, data, idxes):
    with open(saving_directory + '.txt', 'w', encoding='utf-8') as f:
        f.write('frame index dx  dy d\n')
        row_idx = 0
        for row in data:
            row_idx += 1
            f.write(str(row_idx-1) + ' ' + str(idxes[row_idx-1]) + ' ' + ' '.join([str(a) for a in row]) + '\n')


def directCorrection(directory, tag_sym, last_num, features_cnt, algo='FFT'):
    shift_vals = list()
    time_vals = list()
    file_idxes = list()
    shift_vals.append([0, 0])
    # print(os.listdir(directory), os.listdir(directory)[0], len(os.listdir(directory)))
    files = [(os.path.normpath(directory + '/' + f), int(f[tag_sym[0]:tag_sym[1]])) \
             for f in os.listdir(directory) if f.endswith(".tif")]
    sorted_files = sorted(files, key=lambda t: t[1])
    file_idxes = [int(file_name[1]) for file_name in sorted_files]
    last_num = len(sorted_files) if last_num == -1 else last_num
    sorted_files = sorted_files[:last_num + 1]
    print("SORTED FILES!!!!!:::::", sorted_files)
    orig_image = sorted_files[0]
    for im_idx in range(1, len(sorted_files)):
        start_time = time.time()  # засекаем время
        temp_image = orig_image
        orig_image = sorted_files[im_idx]
        img1_eq = cv2.equalizeHist(cv2.imread(temp_image[0], cv2.IMREAD_GRAYSCALE))
        img2_eq = cv2.equalizeHist(cv2.imread(orig_image[0], cv2.IMREAD_GRAYSCALE))

        print('image idx:', orig_image, im_idx + 1, end=" ")
        if algo == 'SIFT':
            shift_idx = SIFT_registration([img1_eq, img2_eq, features_cnt])
        elif algo == 'FFT':
            shift_idx = FFT_registration([img1_eq, img2_eq])
        else:
            print('\n\n\nОШИБКА!!!НЕправильно указан алгоритм регистрации!!!\n\n\n')
            break
        print(shift_idx)
        shift_vals.append(list(shift_idx))

        end_time = time.time()

        # Вычисляем время выполнения участка кода
        execution_time = end_time - start_time
        time_vals.append(execution_time)
        # print(f"Сдвиг: {shift_vals[-1][2]} Время выполнения: {execution_time} секунд")
    print('SUMMARY TIME:', sum(time_vals))
    print('file indexes:', file_idxes)

    print('BEFORE', shift_vals)
    # arr = np.array(shift_vals)
    # mask = np.all(arr < 5, axis=1)
    # arr[mask] = 0
    # cumulative_sum = np.cumsum(arr, axis=0)
    # norms = np.linalg.norm(cumulative_sum[:, :2], axis=1).reshape(-1, 1)
    # result = np.hstack((cumulative_sum, norms))
    # shift_vals = result.tolist()
    # Инициализация нового массива для хранения кумулятивных сумм
    # Вычисление кумулятивной суммы с добавлением текущего элемента к предыдущему
    for i in range(1, len(shift_vals)):
        shift_vals[i][0] = shift_vals[i - 1][0] + shift_vals[i][0]
        shift_vals[i][1] = shift_vals[i - 1][1] + shift_vals[i][1]
        shift_vals[i].append(math.sqrt(shift_vals[i][0]**2 + shift_vals[i][1]**2))
    # Преобразование в список, если это нужно
    print('AFTER', shift_vals)
    # displaying shifted images
    # CT.display_shifted_images(shift_vals, sorted_files)
    print('TIME_VALS:', time_vals, sum(time_vals), sep='\n')
    return shift_vals, time_vals, file_idxes


def directCorrectionParallel(directory, tag_sym, last_num, features_cnt, N_CORES, algo):
    time_vals = list()
    files = [(os.path.normpath(directory + '/' + f), int(f[tag_sym[0]:tag_sym[1]])) \
             for f in os.listdir(directory) if f.endswith(".tif")]
    sorted_files = sorted(files, key=lambda t: t[1])
    # print(sorted_files)
    file_idxes = [int(file_name[1]) for file_name in sorted_files]
    # print('file_idxes:', file_idxes)
    last_num = len(sorted_files) if last_num == -1 else last_num
    sorted_files = sorted_files[:last_num + 1]
    print("N_CORES:", N_CORES)

    start = time.time()
    image_pairs = multi_registration(sorted_files, features_cnt, algo)
    # print(len(image_pairs), image_pairs)
    shift_vals = parallel_process_image_pairs(image_pairs, N_CORES, algo)
    end = time.time()

    time_vals.append(end - start)
    shift_vals.insert(0, (0, 0))
    print(sum(time_vals))

    arr = np.array(shift_vals)
    mask = np.all(arr < 5, axis=1)
    arr[mask] = 0
    cumulative_sum = np.cumsum(arr, axis=0)
    norms = np.linalg.norm(cumulative_sum[:, :2], axis=1).reshape(-1, 1)
    result = np.hstack((cumulative_sum, norms))
    shift_vals = result.tolist()
    print(shift_vals)
    # displaying shifted images
    # CT.display_shifted_images(shift_vals, sorted_files)
    return shift_vals, time_vals, file_idxes


def multi_registration(array, features_cnt, algo):
    pairs = list()

    for i in range(len(array) - 1):
        img1_eq = cv2.equalizeHist(cv2.imread(array[i][0], cv2.IMREAD_GRAYSCALE))
        img2_eq = cv2.equalizeHist(cv2.imread(array[i + 1][0], cv2.IMREAD_GRAYSCALE))
        # img1 = cv2.threshold(img1_eq, 129, 255, cv2.THRESH_BINARY)
        # img2 = cv2.threshold(img2_eq, 129, 255, cv2.THRESH_BINARY)
        if algo=='SIFT':
            print('ARRAY[I]:', array[i][0])
            pair = (img1_eq, img2_eq, features_cnt)
        elif algo=='FFT':
            pair = (img1_eq, img2_eq)
        else:
            print('ОШИБКА!!!НЕПРАВИЛЬНЫЙ АЛГОРИТМ РЕГИСТРАЦИИ!!!')
            break
        pairs.append(pair)

    # print(pairs)
    return pairs


def parallel_process_image_pairs(args, N_CORES, algo):
    num_processes = multiprocessing.cpu_count()  # Получаем количество доступных CPU
    print(f'количество доступных CPU: {num_processes}')
    pool = multiprocessing.Pool(processes=N_CORES)

    if algo=='SIFT':
        results = pool.map(SIFT_registration, args)  # Запускаем обработку пар попарно и получаем результаты
    else:
        results = pool.map(FFT_registration, args)
    pool.close()
    pool.join()

    return results


def parallel_process_image_pairs_opt(image_pairs, n_cores, algo):
    with multiprocessing.Pool(processes=n_cores) as pool:
        chunksize = max(1, len(image_pairs) // n_cores)  # Устанавливаем размер пакета
        # results = pool.map(SIFT_registration, image_pairs, chunksize=chunksize)
        if algo == 'SIFT':
            results = pool.map(SIFT_registration, image_pairs, chunksize=chunksize)  # Запускаем обработку пар попарно и получаем результаты
        else:
            results = pool.map(FFT_registration, image_pairs, chunksize=chunksize)
    return results


def SIFT_registration(args):
    img1_eq, img2_eq, features_cnt = args[0], args[1], args[2]

    # img1_path, img2_path, features_cnt = args[0], args[1], args[2]
    # print("img1_path, img2_path, features_cnt", img1_path, img2_path, features_cnt)
    # img1_eq = cv2.equalizeHist(cv2.imread(img1_path[0], cv2.IMREAD_GRAYSCALE))
    # img2_eq = cv2.equalizeHist(cv2.imread(img2_path[0], cv2.IMREAD_GRAYSCALE))
    img1 = cv2.threshold(img1_eq, 120, 255, cv2.THRESH_BINARY)
    img2 = cv2.threshold(img2_eq, 120, 255, cv2.THRESH_BINARY)

    # cv2.imshow('img1', IP.resizingImage(img1[1], 40))
    # cv2.waitKey()
    # cv2.imshow('img2', IP.resizingImage(img2[1], 40))
    # cv2.waitKey()
    # Инициализация объекта SIFT
    # print('FEATURES COUNT:', features_cnt)
    sift = cv2.SIFT_create(nfeatures=features_cnt)

    # Нахождение ключевых точек и дескрипторов для каждого изображения
    # start = time.time()
    keypoints1, descriptors1 = sift.detectAndCompute(img1[1], None)
    keypoints2, descriptors2 = sift.detectAndCompute(img2[1], None)
    # end = time.time()
    # print(f'sift.detectAndCompute function time:{end - start}')
    # print('Количество ключевых точек на первом снимке:', len(keypoints1), end=" ")
    # print('на втором снимке:', len(keypoints2), end=" ")

    # # Отображение ключевых точек на изображении
    # img1_with_keypoints = cv2.drawKeypoints(img1, keypoints1, None)
    #
    # # Отображение ключевых точек на изображении
    # img2_with_keypoints = cv2.drawKeypoints(img2, keypoints2, None)
    #
    # # Отображение изображения с ключевыми точками
    # cv2.imshow('Image with Keypoints', IP.resizingImage(img1_with_keypoints, 40))
    # cv2.waitKey(0)
    #
    # # Отображение изображения с ключевыми точками
    # cv2.imshow('Image with Keypoints', IP.resizingImage(img2_with_keypoints, 40))
    # cv2.waitKey(0)

    # Нахождение совпадений ключевых точек между изображениями
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(descriptors1, descriptors2, k=2)
    # Применение RANSAC для оценки матрицы трансформации
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    if len(good_matches) > 4:
        src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        # # Применение матрицы трансформации к изображению
        # aligned_img = cv2.warpPerspective(img1, M, (img1.shape[1], img1.shape[0]))

        # #матрица гомографии
        # print('M', M)
        # Извлечение сдвига из матрицы гомографии
        dx = M[0, 2]
        dy = M[1, 2]

        # print("Сдвиг по X:", dx)
        # print("Сдвиг по Y:", dy)
        # print("Норма ||(x, y)||:", math.sqrt(dx**2 + dy**2))
        # sort matches by distance
        # matches = sorted(good_matches, key=lambda x: x.distance)
        #
        # # draw first 50 matches
        # matched_img = cv2.drawMatches(IP.resizingImage(img1, 60), keypoints1, IP.resizingImage(img2, 60), keypoints2, matches[:50], \
        #                               IP.resizingImage(img2, 60), flags=2)
        #
        # # show the image
        # cv2.imshow('image', IP.resizingImage(matched_img, 40))
        # cv2.waitKey(0)

        # Отображение выровненного изображения
        # cv2.imshow('Aligned Image', IP.resizingImage(aligned_img, 40))
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        return dx, dy


def ORB_registration(args):
    img1_path, img2_path, features_cnt = args[0], args[1], args[2]
    img1_eq = cv2.equalizeHist(cv2.imread(img1_path[0], cv2.IMREAD_GRAYSCALE))
    img2_eq = cv2.equalizeHist(cv2.imread(img2_path[0], cv2.IMREAD_GRAYSCALE))
    img1 = cv2.threshold(img1_eq, 129, 255, cv2.THRESH_BINARY_INV)
    img2 = cv2.threshold(img2_eq, 129, 255, cv2.THRESH_BINARY_INV)

    # Создание детектора ORB
    orb = cv2.ORB_create()

    # Нахождение ключевых точек и дескрипторов
    kp1, des1 = orb.detectAndCompute(img1[1], None)
    kp2, des2 = orb.detectAndCompute(img2[1], None)

    # Создание матчера BFMatcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Сопоставление дескрипторов
    matches = bf.match(des1, des2)

    # Сортировка матчей по расстоянию
    matches = sorted(matches, key=lambda x: x.distance)

    mean_shift_x = np.mean([kp2[m.trainIdx].pt[0] - kp1[m.queryIdx].pt[0] for m in matches])
    mean_shift_y = np.mean([kp2[m.trainIdx].pt[1] - kp1[m.queryIdx].pt[1] for m in matches])

    return mean_shift_x, mean_shift_y


def SURF_registration(args):
    img1_path, img2_path, features_cnt = args[0], args[1], args[2]
    img1_eq = cv2.equalizeHist(cv2.imread(img1_path[0], cv2.IMREAD_GRAYSCALE))
    img2_eq = cv2.equalizeHist(cv2.imread(img2_path[0], cv2.IMREAD_GRAYSCALE))
    img1 = cv2.threshold(img1_eq, 129, 255, cv2.THRESH_BINARY_INV)
    img2 = cv2.threshold(img2_eq, 129, 255, cv2.THRESH_BINARY_INV)
    # cv2.imshow('img1', IP.resizingImage(img1[1], 40))
    # cv2.waitKey()
    # cv2.imshow('img2', IP.resizingImage(img2[1], 40))
    # cv2.waitKey()
    # Инициализация объекта SIFT
    surf = cv2.SURF_create()
    # Нахождение ключевых точек и дескрипторов для каждого изображения
    # start = time.time()
    keypoints1, descriptors1 = surf.detectAndCompute(img1[1], None)
    keypoints2, descriptors2 = surf.detectAndCompute(img2[1], None)
    # end = time.time()
    # print(f'sift.detectAndCompute function time:{end - start}')
    # print('Количество ключевых точек на первом снимке:', len(keypoints1), end=" ")
    # print('на втором снимке:', len(keypoints2), end=" ")

    # # Отображение ключевых точек на изображении
    # img1_with_keypoints = cv2.drawKeypoints(img1, keypoints1, None)
    #
    # # Отображение ключевых точек на изображении
    # img2_with_keypoints = cv2.drawKeypoints(img2, keypoints2, None)
    #
    # # Отображение изображения с ключевыми точками
    # cv2.imshow('Image with Keypoints', IP.resizingImage(img1_with_keypoints, 40))
    # cv2.waitKey(0)
    #
    # # Отображение изображения с ключевыми точками
    # cv2.imshow('Image with Keypoints', IP.resizingImage(img2_with_keypoints, 40))
    # cv2.waitKey(0)

    # Нахождение совпадений ключевых точек между изображениями
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(descriptors1, descriptors2, k=2)
    # Применение RANSAC для оценки матрицы трансформации
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    if len(good_matches) > 4:
        src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        # # Применение матрицы трансформации к изображению
        # aligned_img = cv2.warpPerspective(img1, M, (img1.shape[1], img1.shape[0]))

        # #матрица гомографии
        # print('M', M)
        # Извлечение сдвига из матрицы гомографии
        dx = M[0, 2]
        dy = M[1, 2]

        # print("Сдвиг по X:", dx)
        # print("Сдвиг по Y:", dy)
        # print("Норма ||(x, y)||:", math.sqrt(dx**2 + dy**2))
        # sort matches by distance
        # matches = sorted(good_matches, key=lambda x: x.distance)
        #
        # # draw first 50 matches
        # matched_img = cv2.drawMatches(IP.resizingImage(img1, 60), keypoints1, IP.resizingImage(img2, 60), keypoints2, matches[:50], \
        #                               IP.resizingImage(img2, 60), flags=2)
        #
        # # show the image
        # cv2.imshow('image', IP.resizingImage(matched_img, 40))
        # cv2.waitKey(0)

        # Отображение выровненного изображения
        # cv2.imshow('Aligned Image', IP.resizingImage(aligned_img, 40))
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        return dx, dy


def FFT_registration(args):
    # img1_path, img2_path = args[0], args[1]
    #
    # # Преобразование изображений в оттенки серого
    # gray_image1 = cv2.imread(img1_path[0], cv2.IMREAD_GRAYSCALE)
    # gray_image2 = cv2.imread(img2_path[0], cv2.IMREAD_GRAYSCALE)
    #
    # # cv2.imshow("Grayscale image", IP.resizingImage(gray_image1, 40))
    # # cv2.waitKey(0)
    #
    # equ_image1 = cv2.equalizeHist(gray_image1)
    # equ_image2 = cv2.equalizeHist(gray_image2)
    # # cv2.imshow("Equalized image", resizingImage(equ_image, 40))
    # # cv2.waitKey(0)
    equ_image1, equ_image2 = args[0], args[1]
    ret, thresh1 = cv2.threshold(equ_image1, 120, 255, cv2.THRESH_BINARY)
    ret, thresh2 = cv2.threshold(equ_image2, 120, 255, cv2.THRESH_BINARY)

    # cv2.imshow("Threshed image1", IP.resizingImage(thresh1, 40))
    # cv2.waitKey(0)
    # cv2.imshow("Threshed image2", IP.resizingImage(thresh2, 40))
    # cv2.waitKey(0)

    fft_ref = fft2(thresh1)
    fft_target = fft2(thresh2)

    # Вычисляем кросс-корреляцию
    cross_correlation = fft_ref * np.conj(fft_target)
    # print('cross_correlation', cross_correlation)

    cross_correlation_abs = np.abs(cross_correlation)
    cross_correlation_abs[cross_correlation_abs < 1e-10] = 1e-10  # Замена очень маленьких значений на небольшое положительное число
    # Нормализуем кросс-корреляцию
    cross_correlation /= cross_correlation_abs

    # Применяем обратное преобразование Фурье
    shift_before = ifft2(cross_correlation)

    # Определяем смещение
    shift_after = fftshift(shift_before)
    # # Вычисление амплитуды
    # amplitude_spectrum = np.abs(shift_after)
    #
    # # Визуализация амплитудного спектра
    # plt.figure(figsize=(10, 8))
    # plt.imshow(np.log1p(amplitude_spectrum), cmap='gray')
    # plt.title('Spectr')
    # plt.colorbar()
    # plt.show()
    # plt.savefig(r"C:\VKR\Experiments_diplom\reg\10_bad")


    max_idx = np.unravel_index(np.argmax(np.abs(shift_after)), shift_after.shape)

    # Вычисляем сдвиг по осям
    shift_x = -(max_idx[1] - thresh1.shape[1] // 2)
    shift_y = -(max_idx[0] - thresh2.shape[0] // 2)
    print(f'fft shift dx={shift_x}; dy={shift_y}')
    # Возвращаем сдвиг по осям и его норму
    return shift_x, shift_y


def multi_SIFT_registration_JOB(array, features_cnt):
    pairs = list()

    for i in range(len(array) - 1):
        pair = [array[i], array[i + 1], features_cnt]
        pairs.append(pair)

    return list(map(SIFT_registration, pairs))


def SIFTdirectCorrectionParallel_JOB(directory, tag_sym, last_num, features_cnt, N_CORES):
    shift_vals, time_vals = list(), list()
    files = [(os.path.normpath(directory + '/' + f), int(f[tag_sym[0]:tag_sym[1]])) \
             for f in os.listdir(directory) if f.endswith(".tif")]
    sorted_files = sorted(files, key=lambda t: t[1])
    last_num = len(sorted_files) if last_num == -1 else last_num
    sorted_files = sorted_files[:last_num + 1]
    start = time.time()

    list_array = np.array_split(sorted_files, N_CORES)
    for idx in range(1, len(list_array)):
        list_array[idx] = np.vstack([list_array[idx-1][-1], list_array[idx]])
    print(len(list_array))
    data = Parallel(n_jobs=N_CORES, verbose=10)(delayed(multi_SIFT_registration_JOB)(array, features_cnt) for array in list_array)
    for core in data:
        print(len(core))
    shift_vals = reduce(lambda x, y: x.extend(y) or x, data)
    end = time.time()
    time_vals.append(end - start)
    shift_vals.insert(0, (0, 0, 0))
    print(sum(time_vals))
    return shift_vals, time_vals


# def process_image_pair(args):
#     aligned_image_paths, i, features_cnt = args
#
#     # Выравнивание текущего изображения относительно предыдущего
#     dx, dy, displacement = SIFT_registration(aligned_image_paths[-1], aligned_image_paths[i], features_cnt)
#
#     # Путь к выровненному изображению
#     aligned_image_path = f'aligned_image_{i}.jpg'
#
#     print(f'Сдвиг по X для изображения {i}: {dx}')
#     print(f'Сдвиг по Y для изображения {i}: {dy}')
#     print(f'Норма сдвига для изображения {i}: {displacement}')
#
#     return aligned_image_path, dx, dy, displacement
#
#
# def SIFTmultiDirect(directory, tag_sym=(1, 2), last_num=-1, features_cnt=20000):
#     shift_vals, time_vals = list(), list()
#     shift_vals.append([0, 0, 0])
#     # Список путей к изображениям
#     files = [(os.path.normpath(directory + '/' + f), int(f[tag_sym[0]:tag_sym[1]])) \
#              for f in os.listdir(directory) if f.endswith(".tif")]
#     sorted_files = sorted(files, key=lambda t: t[1])
#     last_num = len(sorted_files) if last_num == -1 else last_num
#     sorted_files = sorted_files[:last_num + 1]
#
#     # Первое изображение не нужно выравнивать, поэтому начинаем со второго
#     aligned_image_paths = [sorted_files[0]]
#
#     # Создаем пул процессов
#     start = time.time()
#     with Pool() as pool:
#         # Подготовка аргументов для каждого изображения
#         # args = [(sorted_files, i, features_cnt) for i in range(1, len(sorted_files))]
#         args = [(aligned_image_paths, i, features_cnt) for i in range(1, len(sorted_files))]
#
#         aligned_image_path, dx, dy, displacement = pool.map(process_image_pair, args)
#
#         # Выполняем процесс выравнивания изображений параллельно
#         shift_vals.append([dx, dy, displacement])
#         aligned_image_paths.append([aligned_image_path])
#
#     end = time.time()
#     time_vals.append(end-start)
#     # Вывод путей к выровненным изображениям
#     print("Пути к выровненным изображениям:")
#     for shift_val in shift_vals:
#         print(*shift_val)