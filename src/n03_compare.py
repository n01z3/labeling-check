import pandas as pd
import matplotlib.pyplot as plt
import json
import os
import random
from glob import glob
from multiprocessing import Pool

import cv2
import numpy as np
from tqdm import tqdm
import argparse

from n01_config import get_path
from n02_utils import get_classes, convert_points2poly, iou_numpy, get_spaced_colors2

random.seed(42)
np.random.seed(666)
FONT = cv2.FONT_HERSHEY_SIMPLEX
PATHS = get_path()
COLORS = [(255, 0, 0), (255, 106, 0), (255, 255, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255)]
ALPHA = 0.6


def build_masks(json_filename, klasses):
    with open(json_filename) as f:
        json_data = json.load(f)

    klass2idx = dict([(klass, n) for n, klass in enumerate(klasses)])

    image_filename = os.path.join(PATHS['image_path'], os.path.basename(json_filename).replace('.json', '.jpg'))
    image = cv2.imread(image_filename)
    assert image is not None, f'{image_filename} not exist'
    masks = [np.zeros((image.shape[0], image.shape[1])) for _ in range(len(klasses))]

    if len(json_data['objects']) > 0:
        for el in json_data['objects']:
            klass = el['classTitle'].lower()
            points = el['points']['exterior']
            if klass not in klasses:
                continue
            index = klass2idx.get(klass)
            mask = masks[index].copy()
            cv2.fillPoly(mask, [convert_points2poly(points)], 255)
            masks[index] = mask

    else:
        print(f'empty json file {os.path.basename(json_filename)}')

    return masks


def get_border(mask):
    if np.sum(mask) == 0:
        return 0, 0
    else:
        x_profile = np.nonzero(np.sum(mask, axis=0))[0]
        y_profile = np.nonzero(np.sum(mask, axis=1))[0]
    return x_profile[-1], y_profile[-1]


def put_mask_on_image(image, mask_new, mask_golden, filename):
    labels = ['new', 'golden']
    mask_sources = [mask_new, mask_golden]

    outputs = []
    for z, (label, mask) in enumerate(zip(labels, mask_sources)):
        output = image.copy()
        overlay = image.copy()

        for n, klass in enumerate(KLASSES[:-1]):
            if klass == 'shelf':
                continue
            tmp = mask[n]
            tmp[tmp < 128] = 0
            overlay[tmp > 128] = COLORS[n]

        cv2.addWeighted(overlay, ALPHA, output, 1 - ALPHA, 0, output)
        for n, klass in enumerate(KLASSES):
            cv2.putText(img=output, text=KLASSES[n], org=(10, 60 + 30 * n),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1.0, color=[0, 0, 0],
                        thickness=5)

            cv2.putText(output, KLASSES[n],
                        (10, 60 + 30 * n), cv2.FONT_HERSHEY_SIMPLEX, 1.0, COLORS[n], 2)

        cv2.putText(img=output, text=label, org=(10, 30),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1.0, color=[0, 0, 0],
                    thickness=5)
        cv2.putText(output, label,
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

        outputs.append(output)

    bottom = np.hstack([image, np.zeros(image.shape, np.uint8)])
    result = np.vstack([np.hstack(outputs), bottom])

    cv2.imwrite(filename, result[:, :, ::-1])


def compare_one(golden_filename):
    new_filename = os.path.join(new_dataset_path, os.path.basename(golden_filename))
    golden_masks = build_masks(golden_filename, KLASSES)
    new_masks = build_masks(new_filename, KLASSES)

    borders = np.array([get_border(mask) for mask in new_masks])
    max_borders = (np.amax(borders[:, 1]), np.amax(borders[:, 0]))

    image_basename = os.path.basename(golden_filename).replace('.json', '.jpg')
    image_filename = os.path.join(PATHS['image_path'], image_basename)
    image = cv2.imread(image_filename)[:, :, ::-1]

    if image.shape[0] // 2 >= max_borders[0] and image.shape[1] // 2 >= max_borders[1]:
        new_masks = [cv2.resize(mask, image.shape[:2], interpolation=cv2.INTER_NEAREST) for mask in new_masks]

    out = []
    for new_mask, golden_mask in zip(new_masks, golden_masks):
        klass_iou = iou_numpy(golden_mask, new_mask)
        out.append(klass_iou)

    compare_filename = os.path.join(new_dataset_path, '../compare', image_basename)
    put_mask_on_image(image, new_masks, golden_masks, compare_filename)
    return out


def main():
    global KLASSES, new_dataset_path

    args = parse_args()
    new_dataset_path = args.new_folder

    os.makedirs(os.path.join(new_dataset_path, '../compare'), exist_ok=True)
    golden_filenames = sorted(glob(os.path.join(PATHS['golden_dataset'], '*json')))
    new_filenames = sorted(os.listdir(new_dataset_path))

    KLASSES = get_classes(golden_filenames)
    assert len(golden_filenames) != 0, f'{PATHS["golden_set"]} contain no jsons'

    intersect = [fn for fn in golden_filenames if os.path.basename(fn) in new_filenames]
    intersect = intersect[:10]
    print(f'{len(intersect)} samples for compare')

    if args.multiprocess:
        with Pool() as p:
            metrics = list(tqdm(p.imap(compare_one, intersect), total=len(intersect)))

    else:
        metrics = []
        for golden_filename in tqdm(intersect, total=len(intersect)):
            metrics.append(compare_one(golden_filename))

    metrics = np.array(metrics)

    df = pd.DataFrame()
    df['fns'] = [os.path.basename(fn) for fn in intersect]
    print('average metrics:')
    for n, klass in enumerate(KLASSES):
        df[klass] = metrics[:, n]
        print(f'{klass}:\t{np.mean(metrics[:, n]):0.3f}')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--new_folder', type=str, default='/home/n01z3/dataset/shelf/k_v3/ann')
    parser.add_argument('--multiprocess', type=bool, default=True)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    main()
