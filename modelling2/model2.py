import numpy as np


class Taxel:
    def __init__(self, c0, c1, c2, c3, c4):
        self.c0 = c0
        self.c1 = c1
        self.c2 = c2
        self.c3 = c3
        self.c4 = c4
        self.min = min((c0, c1, c2, c3, c4))
        self.max = max((c0, c1, c2, c3, c4))

    # using the equation y = ax+b, by default no normalization
    def to_matrix(self, a=1, b=0):
        def y(x):
            return a * x + b

        return np.array([
            [0, y(self.c0), 0],
            [y(self.c1), y(self.c2), y(self.c3)],
            [0, y(self.c4), 0],
        ])


class Frame:
    def __init__(self, taxels):
        self.taxels = taxels
        self.min = min([t.min for t in taxels])
        self.max = max([t.max for t in taxels])

    def to_matrix(self, a=1, b=0):
        m = [t.to_matrix(a, b) for t in self.taxels]
        return np.hstack((
            np.vstack((m[0], m[1], m[2], m[3], m[4], m[5])),
            np.vstack((m[6], m[7], m[8], m[9], m[10], m[11])),
            np.vstack((m[12], m[13], m[14], m[15], m[16], m[17])),
            np.vstack((m[18], m[19], m[20], m[21], m[22], m[23])),
            np.vstack((m[24], m[25], m[26], m[27], m[28], m[29])),
            np.vstack((m[30], m[31], m[32], m[33], m[34], m[35])),
        ))


class Gesture_2:
    def __init__(self, frames, label):
        self.frames = frames
        self.label = label
        self.min = min([t.min for t in frames])
        self.max = max([t.max for t in frames])

    def to_matrix(self, a=1, b=0):
        return [f.to_matrix(a, b) for f in self.frames]


class Participant_2:
    def __init__(self, pid, gestures):
        self.pid = pid
        self.gestures = gestures
        self.min = min([t.min for t in gestures.values()])
        self.max = max([t.max for t in gestures.values()])
        print(self.min)
        print(self.max)
