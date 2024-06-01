#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from sys import argv
import glob
import os
import random
import numpy as np
import pandas as pd
from IPython.display import display
# from matplotlib import pyplot as plt, image as mpimg
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from laptrack import LapTrack
import napari
import cv2
from ImageProcessing import resizingImage

plt.rcParams["font.family"] = ""


def csvReading(files, params):
    # Добавление изображения на фон
    img = mpimg.imread(params['bkg_img'])

    if params['custom_offset_mode'] == 'ON':
        spots_df = pd.read_csv(files[0], usecols=['Number', 'xmean', 'ymean'], nrows=params['points_cnt'])
        for idx in range(params['points_cnt'], params['frames_cnt'] * params['points_cnt']):
            dx, dy = random.randint(-params['offset'], params['offset']), random.randint(-params['offset'],
                                                                                         params['offset'])
            newX, newY = spots_df.iloc[idx - params['points_cnt']]['xmean'] + dx, \
                         spots_df.iloc[idx - params['points_cnt']]['ymean'] + dy
            spots_df.loc[len(spots_df.index)] = [idx % params['points_cnt'], newX, newY]
        spots_df.insert(0, "frame", [(i - (i % params['frames_cnt'])) // params['frames_cnt'] for i in\
                                     range(params['frames_cnt'] * params['points_cnt'])], True)
    else:
        li = []
        i = 0
        for filename in files:
            print(filename)
            tmp_df = pd.read_csv(filename, usecols=['Number', 'xmean', 'ymean', 'm000'])
            tmp_df.insert(0, "frame", i, True)

            print(len(tmp_df))
            li.append(tmp_df)
            i += 1
        spots_df = pd.concat(li, axis=0, ignore_index=True)
        # Select colonies by area
        lower_bound, upper_bound = params['sq_lower_bound'], params['sq_upper_bound']
        #сортировка ТОЛЬКО первого кадра датафрейма по площади колоний
        spots_df = spots_df[~((spots_df['frame'] == 0) & (~spots_df['m000'].between(lower_bound, upper_bound)))]
        # сортировка ВСЕХ кадров датафрейма по площади колоний
        # spots_df = spots_df[spots_df['m000'].between(lower_bound, upper_bound)]
        display(spots_df)
        spots_df = spots_df.drop('m000', axis=1)

    spots_df['Number'] = [int(n) for n in spots_df['Number']]
    spots_df['ymean'] = img.shape[0]-spots_df['ymean']
    # print('img.shape[1]', img.shape[0])
    # for point_idx in range(params['points_cnt']):
    #     print("\n Все кадры для точки №", point_idx)
    #     display(spots_df[spots_df.index % params['points_cnt'] == point_idx])

    display(spots_df)

    return spots_df


def csvWriting(track_df, split_df, merge_df, saving_path):
    print('Saving path: ', saving_path)
    if not os.path.exists(saving_path):
        print('Creating directory for saving...')
        os.mkdir(saving_path)

    track_df.to_csv(os.path.join(saving_path,'track_df.csv'))
    split_df.to_csv(os.path.join(saving_path,'split_df.csv'))
    merge_df.to_csv(os.path.join(saving_path,'merge_df.csv'))


def csvHighlighting(directory, mask, tag_sym):
    # Получаем список всех файлов в директории, соответствующих маске
    # 11:15, чтоб индекс забирать, надо это учитвать при названии файла

    files = [(os.path.normpath(directory + '/' + f), int(f[tag_sym[0]:tag_sym[1]])) for f in os.listdir(directory) if
                  f.endswith(mask)]
    # print(files)

    # Сортировка
    sorted_files = sorted(files, key=lambda t: t[1])
    # print(sorted_files)

    return sorted_files


def LapTrackCore(spots_df, params):
    lt = LapTrack(
        track_dist_metric="sqeuclidean",
        # The similarity metric for particles. See `scipy.spatial.distance.cdist` for allowed values.
        splitting_dist_metric="sqeuclidean",
        merging_dist_metric="euclidean",
        # the square of the cutoff distance for the "sqeuclidean" metric
        track_cost_cutoff=params['max_dist_param']**2,
        # track_start_cost=-1,
        splitting_cost_cutoff=False,  # or False for non-splitting case
        merging_cost_cutoff=(params['max_dist_param'])*params['merging_cost_cutoff_multiplier'],  # or False for non-merging case
        gap_closing_max_frame_count=params['max_gap'],
        track_start_cost=9999,
        track_end_cost=9999
        # merging_cost_cutoff=False,  # or False for non-merging case
    )

    track_df, split_df, merge_df = lt.predict_dataframe(
        spots_df,
        coordinate_cols=[
            "xmean",
            "ymean",
        ],  # the column names for the coordinates
        frame_col="frame",  # the column name for the frame (default "frame")
        only_coordinate_cols=False,  # if False, returned track_df includes columns not in coordinate_cols.
        # False will be the default in the major release.
    )

    keys = ["xmean", "ymean", "track_id", "tree_id"]

    print("TrackDataFrame: \n", track_df)
    display(track_df[keys].head())

    print('SPLIT_DF')
    display(split_df)

    print('MERGE_DF')
    display(merge_df)

    return track_df, split_df, merge_df, keys


def NapariViewer(track_df, spots_df):
    v = napari.Viewer()
    v.add_points(spots_df[["frame", "xmean", "ymean"]])
    track_df2 = track_df.reset_index()
    v.add_tracks(track_df2[["track_id", "frame", "xmean", "ymean"]])
    v.show()


def get_track_end(track_df, track_id, keys, first=True):
    df = track_df[track_df["track_id"] == track_id].sort_index(level="frame")
    return df.iloc[0 if first else -1][keys]


def filter_tracks(track_df, merge_df, track_len):
    tmp_track_df = pd.DataFrame()  # Создаем пустой датафрейм для хранения отфильтрованных данных
    filtered_track_df = pd.DataFrame()

    for track_id, grp in track_df.groupby("track_id"):
        df = grp.reset_index().sort_values("frame")
        if len(df) > 2:
            if df['frame'].iloc[0] == 0:
                tmp_track_df = pd.concat(
                    [tmp_track_df, df])

    for tree_id, grp in tmp_track_df.groupby("tree_id"):
        # print('group_by_tree_id \n', grp)
        df = grp.reset_index().sort_values("frame")
        print('grp_df \n', df)
        print('Последний кадр:', df['frame'].iloc[-1], type(df['frame'].iloc[-1]))
        # if (df['frame'].iloc[-1] - df['frame'].iloc[0]) > track_len:
        if len(df) > track_len:
            filtered_track_df = pd.concat(
                [filtered_track_df, df])  # Добавляем отфильтрованные данные в новый датафрейм

    # Сбрасываем индексы в новом датафрейме
    print('filtered \n', filtered_track_df)

    # Создаем пустой датафрейм для хранения отфильтрованных данных из merge_df
    filtered_merge_df = pd.DataFrame()

    # Оставляем только записи из merge_df, для которых значения столбца parent_track_id есть в filtered_track_df['track_id']
    # Список для хранения строк, которые будут добавлены
    rows_to_add = []

    # Проход по строкам merge_df
    for index, row in merge_df.iterrows():
        if (row['parent_track_id'] in filtered_track_df['track_id'].values) and (
                row['child_track_id'] in filtered_track_df['track_id'].values):
            rows_to_add.append(row)

    # Конкатенация строк к filtered_merge_df
    if rows_to_add:
        filtered_merge_df = pd.concat([filtered_merge_df, pd.DataFrame(rows_to_add)], ignore_index=True)

    # Сбрасываем индексы в новом датафрейме
    filtered_merge_df = filtered_merge_df.reset_index(drop=True)

    # Возвращаем отфильтрованные датафреймы filtered_track_df и filtered_merge_df
    print('filtered_track_df \n', filtered_track_df)
    print('filtered_merge_df \n', filtered_merge_df)
    return filtered_track_df, filtered_merge_df


def MatPlotLibGraphics(track_df, split_df, merge_df, background_image_path, frame_range, split_merge, save_image_path):
    plt.figure(figsize=(3, 3))
    # frames = [track_df['frame'].min(), track_df['frame'].max()]
    print('FRAME RANGE:', frame_range[0], frame_range[1])
    # frame_range = [frames[0], frames[1]]
    k1, k2 = "xmean", "ymean"
    keys = [k1, k2]

    for track_id, grp in track_df.groupby("track_id"):
        # for track_id OR tree_id
        df = grp.sort_values("frame")
        plt.scatter(df[k1], df[k2], c=df["frame"], vmin=frame_range[0], vmax=frame_range[1])

        plt.annotate(str(df['tree_id'].iloc[0]), (df[k1].iloc[0], df[k2].iloc[0]), textcoords="offset points",\
                     xytext=(0, 11), ha='center', color="green")
        # Добавление аннотаций с track_id для каждого сегмента траектории
        # for i, row in df.iterrows():
        #     plt.annotate(str(track_id), (row[k1], row[k2]), textcoords="offset points", xytext=(0, 10), ha='center',
        #                  color="green")

        for i in range(len(df) - 1):
            pos1 = df.iloc[i][keys]
            pos2 = df.iloc[i + 1][keys]
            plt.plot([pos1.iloc[0], pos2.iloc[0]], [pos1.iloc[1], pos2.iloc[1]], "-r")

        if split_merge == 'True':
            for _, row in list(split_df.iterrows()) + list(merge_df.iterrows()):
                pos1 = get_track_end(track_df, row["parent_track_id"], keys, first=False)
                pos2 = get_track_end(track_df, row["child_track_id"], keys, first=True)
                plt.plot([pos1.iloc[0], pos2.iloc[0]], [pos1.iloc[1], pos2.iloc[1]], "-c")

    plt.colorbar(label='Frame Number')
    plt.xticks([])
    plt.yticks([])

    # Добавление изображения на фон
    img = mpimg.imread(background_image_path)
    flipped_img = img
    plt.imshow(flipped_img, extent=[0, img.shape[1], 0, img.shape[0]], cmap='gray')
    plt.show()
    plt.savefig(save_image_path)


def MatPlotLibGraphics_OLD(track_df, split_df, merge_df, background_image_path, track_len, split_merge):
    plt.figure(figsize=(3, 3))
    frames = track_df.index.get_level_values("frame")
    frame_range = [frames.min(), frames.max()]
    k1, k2 = "xmean", "ymean"
    keys = [k1, k2]

    print('track_df', track_df)

    for track_id, grp in track_df.groupby("track_id"):
        if len(grp) > track_len:
            df = grp.reset_index().sort_values("frame")
            if df['frame'].iloc[0] == 0:
                print('Траектории, начинающиеся с первого кадра.')
                display(df)

                plt.scatter(df[k1], df[k2], c=df["frame"], vmin=frame_range[0], vmax=frame_range[1])

                # Добавление аннотаций с track_id
                for i, row in df.iterrows():
                    plt.annotate(str(track_id), (row[k1], row[k2]), textcoords="offset points", xytext=(0, 10),\
                                 ha='center', color="green")

                for i in range(len(df) - 1):
                    pos1 = df.iloc[i][keys]
                    pos2 = df.iloc[i + 1][keys]
                    plt.plot([pos1.iloc[0], pos2.iloc[0]], [pos1.iloc[1], pos2.iloc[1]], "-r")
                    # plt.scatter(pos1.iloc[0], pos1.iloc[1], c='cyan', s=200)
                    # plt.scatter(pos2.iloc[0], pos2.iloc[1], c='cyan', s=100)
                if split_merge == 'True':
                    for _, row in list(split_df.iterrows()) + list(merge_df.iterrows()):
                        pos1 = get_track_end(track_df, row["parent_track_id"], keys, first=False)
                        pos2 = get_track_end(track_df, row["child_track_id"], keys, first=True)
                        plt.plot([pos1.iloc[0], pos2.iloc[0]], [pos1.iloc[1], pos2.iloc[1]], "-r")

    plt.colorbar(label='Frame Number')
    plt.xticks([])
    plt.yticks([])

    # Добавление изображения на фон
    img = mpimg.imread(background_image_path)
    # rotated_img = np.rot90(img)
    # flipped_img = np.fliplr(img)
    flipped_img = img
    plt.imshow(flipped_img, extent=[0, img.shape[1], 0, img.shape[0]], cmap='gray')
    plt.show()


def consoleInterface():
    # current params: script, path, points_cnt, offset, max_dist_param, frames_cnt(const = all_files), napari_viewer,
    # CustomOffsetMode

    paramsDict = {'script': 'C:\\Users\\xjisc\\PycharmProjects\\LapTrackTest\\CoreTrackingAPI.py',
                  'reading_path': 'C:\\ВКР\Coordinates2Tracks\\first10files', 'saving_path': 'C:\\ВКР\Coordinates2Tracks', \
                  'offset': '10', 'max_dist_param': '10',
                  'points_cnt': '10', 'napari_viewer': 'ON', 'custom_offset_mode': 'ON',
                  'file_mask': '.csv', 'max_gap': 2, 'sq_lower_bound': 999, 'sq_upper_bound': 100000, \
                  'tag_sym': '11,15', 'bkg_img': 'W0001F0002T0001Z001C1.tif', 'split_merge': False, 'min_track_len': 2,\
                  'shifts_path': 'C:\\VKR\\testParallel\\36shifted_old\\result_FFT.txt',\
                  'merging_cost_cutoff_multiplier':1}
    params = argv
    paramsDict['script'] = params[0]
    paramsDict['reading_path'] = params[1]
    paramsDict['saving_path'] = params[2]

    for i in range(3, len(params)):
        paramsDict[params[i].split('=')[0]] = params[i].split('=')[1]

    all_files = glob.glob(paramsDict['reading_path'] + "/" + paramsDict['file_mask'])  # все файлы рассматриваемой директории
    print("Размер директории: ", len(all_files))

    print(paramsDict['tag_sym'])
    print(paramsDict['tag_sym'][1])
    desired_files = csvHighlighting(paramsDict['reading_path'], paramsDict['file_mask'],
                                         tag_sym=list(map(int,paramsDict['tag_sym'].split(","))))
    print(paramsDict['tag_sym'][1])

    paramsDict['frames_cnt'] = len(desired_files)
    # в базовом состоянии равен количеству файлов в рассматриваемой директории, но можно сделать
    # гиперпараметром
    paramsDict['points_cnt'] = int(paramsDict['points_cnt'])  # гиперпараметр, который указываем при запусе скрипта
    paramsDict['offset'] = int(paramsDict['offset'])
    paramsDict['max_dist_param'] = int(paramsDict['max_dist_param'])
    paramsDict['sq_lower_bound'] = int(paramsDict['sq_lower_bound'])
    paramsDict['sq_upper_bound'] = int(paramsDict['sq_upper_bound'])
    paramsDict['max_gap'] = int(paramsDict['max_gap'])
    paramsDict['min_track_len'] = int(paramsDict['min_track_len'])
    paramsDict['merging_cost_cutoff_multiplier'] = float(paramsDict['merging_cost_cutoff_multiplier'])

    print("Этот скрипт называется: ", paramsDict['script'])
    print("Путь к директории с данными: ", paramsDict['reading_path'])
    print("Маска файла с данными: ", paramsDict['file_mask'])
    print("Количество точек: ", paramsDict['points_cnt'])
    print("Количество кадров(const): ", paramsDict['frames_cnt'])
    print("Интервал сдвига: ", "(", -paramsDict['offset'], ", ", paramsDict['offset'], ")", sep='')
    print("Максимальное расстояние: ", paramsDict['max_dist_param'])
    print("Максимальное количество кадров для пропуска:", paramsDict['max_gap'])
    print("NapariViewer: ", paramsDict['napari_viewer'])
    print("CustomOffsetMode: ", paramsDict['custom_offset_mode'])
    print("Нижняя и верхняя границы для площади колоний молекул:", paramsDict['sq_lower_bound'], \
          paramsDict['sq_upper_bound'])
    print("Отрисовка слияний и разделений:", paramsDict['split_merge'])
    print("Минимальная длина трека при отрисовке:", paramsDict['min_track_len'])
    print("Путь к текстовому файлу со сдвигами:", paramsDict['shifts_path'])
    print("Множитель для границы стоимости слияния:", paramsDict['merging_cost_cutoff_multiplier'])

    return desired_files, paramsDict


def coordinates_correction(df, shifts_path):
    # print("ПУТЬ ДО СДВИГОВ:", shifts_path)
    pd.set_option('display.max_rows', 500)
    selected_cols = ['frame', 'index', 'dx', 'dy']
    df_updates = pd.read_csv(shifts_path, delim_whitespace=True, usecols=selected_cols)
    print(df_updates.iloc[0])
    print(df_updates)
    print(df)

    merged_df = df.merge(df_updates, on='frame')
    print(merged_df)
    # display(df.loc[df['frame'] == 6])

    merged_df['xmean'] -= merged_df['dx']
    merged_df['ymean'] += merged_df['dy']

    final_df = merged_df.drop(columns=['dx', 'dy', 'index'])
    print('Updated dataframe:')
    print(final_df)

    return final_df


def apply_shift(image_path, shift):
    # Загружаем изображение
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    equ_image = cv2.equalizeHist(img)
    # cv2.imshow("Equalized image", resizingImage(equ_image, 40))
    # cv2.waitKey(0)

    ret, thresh = cv2.threshold(equ_image, 127, 255, cv2.THRESH_BINARY_INV)
    # Создаем матрицу трансформации
    M = np.float32([[1, 0, -shift[0]], [0, 1, -shift[1]]])
    # Применяем аффинное преобразование
    shifted_img = cv2.warpAffine(thresh, M, (thresh.shape[1], thresh.shape[0]))
    return shifted_img


def display_shifted_images(shifts, image_paths):
    shifted_images = [apply_shift(path[0], shift) for path, shift in zip(image_paths, shifts)]
    idx=0
    for shifted_image in shifted_images:
        cv2.imshow(f"Shifted image {idx}", resizingImage(shifted_image, 40))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        idx+=1


def Core():
    # ConsoleInterface
    _all_files, _paramsDict = consoleInterface()

    # ReadingData
    print([item[0] for item in _all_files])
    spots_df = csvReading([item[0] for item in _all_files], _paramsDict)

    spots_df = coordinates_correction(spots_df, _paramsDict['shifts_path'])
    # Tracking
    track_df, split_df, merge_df, keys = LapTrackCore(spots_df, _paramsDict)
    # print('19 track_id \n', track_df[track_df['track_id'] == 19])
    # print('7 track_id \n', track_df[track_df['track_id'] == 7])
    # print('68 track_id \n', track_df[track_df['track_id'] == 68])
    # print('17 tree_id \n', track_df[track_df['tree_id'] == 17])

    frames = track_df.index.get_level_values("frame")
    frame_range = [frames.min(), frames.max()]
    # Filtering
    # Фильтрация траекторий
    # можно добавить и фильтрацию для split_df, но мы не рассматривавем возможность разделения траекторий в силу
    # специфичности эксперимента
    track_df, merge_df = filter_tracks(track_df, merge_df, _paramsDict['min_track_len'])

    # Graphic part
    if _paramsDict['napari_viewer'] == 'ON':
        NapariViewer(track_df, spots_df)

    MatPlotLibGraphics(track_df, split_df, merge_df, _paramsDict['bkg_img'], frame_range,\
                       _paramsDict['split_merge'], _paramsDict['save_image_path'])
                   #    r'C:\ВКР\TEST_28_01_24\new_data\W0006F0007T0001Z001C1.tif')
    #Writing
    csvWriting(track_df.drop('frame_y', axis=1), split_df, merge_df, _paramsDict['saving_path'])


if __name__ == '__main__':
    Core()

# actual on 24.11 ex:
# python C:\GitLabRepositories\colonies_tracking_base\ColoniesTracker\CoreTrackingAPI.py C:\ВКР\Coordinates2Tracks\first10files C:\ВКР\Coordinates2Tracks\result1 offset=10 max_dist_param=10 points_cnt=10 napari_viewer=ON custom_offset_mode=ON
# python C:\GitLabRepositories\colonies_tracking_base\ColoniesTracker\CoreTrackingAPI.py \
# C:\ВКР\TEST_28_01_24\colony_res_28_01\tab_csv_files_fresh\10 C:\ВКР\TEST_28_01_24\colony_res_28_01\tab_csv_files_fresh\10\result \
# offset=10 max_dist_param=150 points_cnt=10 napari_viewer=ON custom_offset_mode=OFF file_mask=_tab.csv sq_lower_bound=9999\
# sq_upper_bound=19999 max_gap=3