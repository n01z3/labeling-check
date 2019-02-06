import numpy as np
import json
import os
import random
from glob import glob
from multiprocessing import Pool

import cv2
import numpy as np
from tqdm import tqdm


def iou_numpy(outputs: np.array, labels: np.array):
    component1 = np.array(outputs > 0.5 * np.amax(outputs), dtype=bool)
    component2 = np.array(labels > 0.5, dtype=bool)

    if np.sum(component1) == 0 and np.sum(component2) == 0:
        return 1.

    overlap = component1 * component2  # Logical AND
    union = component1 + component2  # Logical OR

    iou = overlap.sum() / float(union.sum())
    return iou


def get_spaced_colors2(n):
    r, g, b = [int(random.random() * 256) for _ in range(3)]
    step = 256 / n
    ret = []
    for i in range(n):
        r += step
        g += step
        b += step
        ret.append((int(r) % 256, int(g) % 256, int(b) % 256))
    return ret


def get_classes(json_filenames):
    classes = []
    for filename in json_filenames:
        with open(filename) as f:
            json_data = json.load(f)
            for el in json_data['objects']:
                classes.append(el['classTitle'].lower())
        f.close()
    return sorted(set(classes))


def convert_points2poly(pts):
    if len(pts) == 2:
        p1 = np.array(pts[0], dtype=int)
        p2 = np.array(pts[1], dtype=int)
        return np.array([(p1[0], p1[1]), (p1[0], p2[1]), (p2[0], p2[1]), (p2[0], p1[1])])

    return np.array([(int(el[0]), int(el[1])) for el in pts])
