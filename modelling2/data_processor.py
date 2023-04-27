import glob
import os

import numpy as np
import pandas as pd

from model import Participant
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
            gesture, avg = raw_data_to_gestures(df)
            gestures.update(gesture)
            all_avg.extend(avg)

        participants.append(Participant(i, gestures))
    with open('avg.npy', 'wb') as f:
        np.save(f, np.asarray(all_avg))
        f.flush()
    return participants


def main(folder, participant_nums, type):
    participants = read_data(participant_nums, type)
    labels = ascii_to_gesture.values()
    to_write = dict()
    for l in labels:
        to_write[l] = []
    for p in participants:
        for g in p.gestures.values():
            g.minmaxvals = p.minmax
            to_write[g.label].append(g)
    os.mkdir(folder)
    for label, gestures in to_write.items():
        os.mkdir(f"{folder}/{label}")
        print("working on " + label)
        for i, g in enumerate(gestures):
            try:
                for j, sample in enumerate(g.chop()):
                    to_video(f"{folder}/{label}/{i}_{j}", sample, g.minmaxvals)
            except Exception:
                print(f"chop or video conversion failed for {label},{i}_{j}")


if __name__ == "__main__":
    # to run all participants use range(1, 14) for second parameter
    main("video_data_1", [1], "FLAT")
