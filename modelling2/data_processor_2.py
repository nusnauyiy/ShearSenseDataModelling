import glob
import os
from random import random, sample

import numpy as np
import pandas as pd

from model import Participant
from model2 import Taxel, Gesture_2, Participant_2, Frame
from util import raw_data_to_gestures, to_video, ascii_to_gesture


# type is one of PILLOW or FLAT
def read_data(participant_nums, type):
    cwd = os.getcwd()
    participants = []
    all_avg = []
    for i in participant_nums:
        all_flat = sorted(glob.glob(os.path.join(cwd, f"../data/P{i}_*{type}*csv")))
        #     all_pillow = sorted(glob.glob(os.path.join(cwd, f"data/P{i}*PILLOW*.csv")))
        gestures = {}
        for f in sorted(all_flat):
            df = pd.read_csv(f, header=None, skiprows=[0], on_bad_lines='skip')
            gesture, avg = process_raw_data(df)
            gestures.update(gesture)
            all_avg.extend(avg)

        participants.append(Participant_2(i, gestures))
    return participants


pressure_idx = list(filter(lambda n: n % 5 == 2, range(0, 180)))
shear_up_idx = list(filter(lambda n: n % 5 == 0, range(0, 180)))
shear_down_idx = list(filter(lambda n: n % 5 == 4, range(0, 180)))
shear_left_idx = list(filter(lambda n: n % 5 == 1, range(0, 180)))
shear_right_idx = list(filter(lambda n: n % 5 == 3, range(0, 180)))


def process_raw_data(df):
    all_avg = []

    def decompose_data(name, df_np, avg):
        def get_delta_c(idx):
            raw_count = df_np[:,idx]
            avg_tile = np.tile(avg[idx], (raw_count.shape[0], 1))
            return (raw_count - avg_tile)

        # convert labels to respective gestures
        label = ascii_to_gesture[name]

        # convert all raw count to delta C
        c0 = get_delta_c(shear_up_idx)
        c1 = get_delta_c(shear_left_idx)
        c2 = get_delta_c(pressure_idx)
        c3 = get_delta_c(shear_left_idx)
        c4 = get_delta_c(shear_down_idx)

        frames = []
        for i, arr in enumerate(c0):
            taxels = []
            for j, _ in enumerate(arr):
                taxels.append(Taxel(c0[i, j], c1[i, j], c2[i, j], c3[i, j], c4[i, j]))
            frames.append(Frame(taxels))
        return Gesture_2(frames, label)

    # average of the first 100 rows
    avg = df.iloc[0, 1:181].to_numpy()
    all_avg.append(avg)

    # remove unlablled data
    df = df[df[182] != 0]
    grouped_df = df.groupby(df.columns[-1])
    gestures = {}
    for name, group in grouped_df:
        try:
            gesture = decompose_data(name, group.iloc[1:, 1:181].to_numpy(), avg)
            gestures[gesture.label] = gesture
        except Exception as e:
            print(e)
            pass

    return gestures, all_avg


def chop(matrix, length=120, num_samples=100):
    adjusted_num_samples = min(num_samples, len(matrix) - length)
    print(adjusted_num_samples)
    samples = sample(range(0, len(matrix) - length), adjusted_num_samples)
    answer = []
    for s in samples:
        answer.append(matrix[s:s + length])
    return answer


def main(folder, participant_nums, type):
    os.mkdir(folder)
    participants = read_data(participant_nums, type)
    counter = 00
    for p in participants:
        for key, val in p.gestures.items():
            isExist = os.path.exists(f"{folder}/{key}")
            if not isExist:
                # Create a new directory because it does not exist
                os.makedirs(f"{folder}/{key}")
            try:
                m = val.to_matrix(a=255.0/(100+700), b=-700)
                for c in chop(m):
                    to_video(f"{folder}/{key}/{counter}", c, width=180, height=180)
                    counter += 1
            except Exception:
                print(f"chop or video conversion failed for {key},{counter}")
    # labels = ascii_to_gesture.values()
    # to_write = dict()
    # for l in labels:
    #     to_write[l] = []
    # for p in participants:
    #     for g in p.gestures.values():
    #         g.minmaxvals = p.minmax
    #         to_write[g.label].append(g)
    # os.mkdir(folder)
    # for label, gestures in to_write.items():
    #     os.mkdir(f"{folder}/{label}")
    #     print("working on " + label)
    #     for i, g in enumerate(gestures):
    #         try:
    #             for j, sample in enumerate(g.chop()):
    #                 to_video(f"{folder}/{label}/{i}_{j}", sample, g.minmaxvals)
    #         except Exception:
    #             print(f"chop or video conversion failed for {label},{i}_{j}")


if __name__ == "__main__":
    # to run all participants use range(1, 14) for second parameter
    main("video_data_D", range(1,13), "FLAT")
