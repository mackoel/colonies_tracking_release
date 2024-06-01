import math

import ImageRegistration as IR
import ImageProcessing as IP
import numpy as np
import argparse
import imutils
import cv2
from sys import argv
import glob


def consoleInterface():
    paramsDict = {'script': 'C:\\GitLabRepositories\\colonies_tracking_base\\ColoniesTracker\\ImageRegistrationCore.py',
                  'reading_path': 'C:\\VKR\\testParallel',
                  'saving_path': 'C:\\VKR\\testParallel\\result',
                  'viewer': 'ON', 'tag_sym': '1,2', 'last_num': -1, 'features_cnt': 20000, 'N_CORES': 4,
                  'parallel': 'ON', 'reg_algo':'SIFT'
                  }
    params = argv
    paramsDict['script'] = params[0]
    paramsDict['reading_path'] = params[1]
    paramsDict['saving_path'] = params[2]

    for i in range(3, len(params)):
        paramsDict[params[i].split('=')[0]] = params[i].split('=')[1]

    all_files = glob.glob(paramsDict['reading_path'] + "/*.tif")  # все файлы рассматриваемой директории
    print("Размер директории: ", len(all_files))

    paramsDict['frames_cnt'] = len(
        all_files)  # в базовом состоянии равен количеству файлов в рассматриваемой директории, но можно сделать
                    # гиперпараметром

    print("Этот скрипт называется: ", paramsDict['script'])
    print("Путь к директории с данными: ", paramsDict['reading_path'])
    print("Путь к директории для сохранения данных: ", paramsDict['saving_path'])
    print("Количество фреймов в директории: ", paramsDict['frames_cnt'])
    print("Viewer: ", paramsDict['viewer'])
    print("Количество ключевых точек: ", paramsDict['features_cnt'])
    print("tag sym: ", paramsDict['tag_sym'])
    print("last num:", paramsDict['last_num'])
    print("Использование параллельной регистрации изображений:", paramsDict['parallel'])
    print("Количество ядер для параллельной регистрации изображений:", paramsDict['N_CORES'])
    print("Алгоритм регистрации изображений:", paramsDict['reg_algo'])

    return paramsDict


def example_pair_correction():
    paramsDict = consoleInterface()
    # filename_0 = "C:\GitLabRepositories\colonies_tracking_base\ColoniesTracker\scans\\new_data\W0006F0007T0001Z001C1.tif"
    # filename_1 = "C:\GitLabRepositories\colonies_tracking_base\ColoniesTracker\scans\\new_data\W0006F0007T0002Z001C1.tif"
    # filename_0 = "C:\GitLabRepositories\colonies_tracking_base\ColoniesTracker\scans\96.tifclass_1.tif"
    # filename_1 = "C:\GitLabRepositories\colonies_tracking_base\ColoniesTracker\scans\97.tifclass_1.tif"
    filename_0 = r"C:\VKR\Experiments_diplom\reg\10_bad\W0001F0001T0096Z001C1.tifclass_1.tif"
    filename_1 = r"C:\VKR\Experiments_diplom\reg\10_bad\W0001F0001T0097Z001C1.tifclass_1.tif"

    template = IP.imageOpening(filename_0)
    original = IP.imageOpening(filename_1)
    print(type(original))
    # print('read dimensions')
    # height, width, channels = template.shape
    # print(f"Количество каналов в изображении: {channels}")
    # cv2.imshow('Read template image.', IP.resizingImage(template, 40))
    # cv2.waitKey(0)
    # cv2.imshow('Read original image.', IP.resizingImage(original, 40))
    # print('read dimensions ended')
    cv2.waitKey(0)

    template_mod = IP.processing(template)
    original_mod = IP.processing(original)
    # template_mod,  original_mod = IP.processing(template, original)

    corrected_image, shift_val = IR.pairCorrection(template_mod, original_mod,
                                                   cv2.cvtColor(np.array(original), cv2.COLOR_RGB2BGR))
    print('shift-vector between images z = (x, y): ', 'dx:', shift_val[0, 0], ', ', 'dy:', shift_val[0, 1], ', ',\
          "||z||=", math.sqrt(shift_val[0, 0]**2 + shift_val[0, 1]**2), sep='')
    cv2.imshow('resTemplate', IP.resizingImage(corrected_image, 40))
    cv2.waitKey(0)


def core():
    # path = 'C:\\ВКР\\CellTracking\\DATA_PHOTO\\TEST2_FULL\\TEST2_DEFAULT_batch'
    # saving_directory = 'C:\\ВКР\\CellTracking\\DATA_PHOTO\\TEST2_FULL\\TEST2_DEFAULT_batch\\shift_for_each_photo_relative_to_the_first_image'
    paramsDict = consoleInterface()
    shift_vals, tim_vals, file_idxes = IR.SPOTdirectCorrection(paramsDict['reading_path'],\
                               list(map(int,paramsDict['tag_sym'].split(","))))
    print(shift_vals, type(shift_vals), len(shift_vals))
    # for idx in range(1, len(shift_vals) - 1):
    #     print('shift_vals', shift_vals[idx])
    #     if (shift_vals[idx][0] == 0) and (shift_vals[idx][1] == 0):
    #         # print("IDX:", idx)
    #         # print("value:", shift_vals[idx - 1])
    #         # print("value:", shift_vals[idx])
    #         # print("value:", shift_vals[idx + 1])
    #         shift_vals[idx][0] = int(0.5*(shift_vals[idx - 1][0] + shift_vals[idx + 1][0]))
    #         shift_vals[idx][1] = int(0.5*(shift_vals[idx - 1][1] + shift_vals[idx + 1][1]))
    #         print("value:", shift_vals[idx])

    # bebe = [shift_vals[q][0] for q in range(len(shift_vals))]
    # print(len(bebe), bebe)
    # for i in range(len(bebe)):
    #     bebe[i].append(math.sqrt(bebe[i][0]**2 + bebe[i][1]**2))
    # Преобразование списка массивов в один numpy массив
    print('SHIFTafdsgdsfs', shift_vals)
    for i in range(1, len(shift_vals)):
        shift_vals[i][0] = shift_vals[i - 1][0] + shift_vals[i][0]
        shift_vals[i][1] = shift_vals[i - 1][1] + shift_vals[i][1]
        shift_vals[i].append(math.sqrt(shift_vals[i][0]**2 + shift_vals[i][1]**2))
    print('SHIFT_VALS FINAL:', shift_vals)
    IR.writingShifts(saving_directory=paramsDict['saving_path'], data=shift_vals, idxes=file_idxes)


def SIFT_core():
    paramsDict = consoleInterface()
    if paramsDict['parallel'] == 'ON':
        shift_vals, time_vals, file_idxes = IR.directCorrectionParallel(paramsDict['reading_path'],\
            tag_sym=list(map(int,paramsDict['tag_sym'].split(","))), last_num=int(paramsDict['last_num']),\
            features_cnt=int(paramsDict['features_cnt']), N_CORES=int(paramsDict['N_CORES']),\
            algo=paramsDict['reg_algo'])
    else:
        shift_vals, time_vals, file_idxes = IR.directCorrection(paramsDict['reading_path'], \
            tag_sym=list(map(int, paramsDict['tag_sym'].split(","))), last_num=int(paramsDict['last_num']),\
            features_cnt=int(paramsDict['features_cnt']), algo=paramsDict['reg_algo'])

    print(shift_vals)
    print(len(shift_vals))
    IR.writingShifts(saving_directory=paramsDict['saving_path'], data=shift_vals, idxes=file_idxes)


if __name__ == '__main__':
    # core()
    SIFT_core()
    # example_pair_correction()
    # example:
