import os
import sys
sys.path.append('.')
sys.path.append('../..')
import numpy as np
from data_generator import input_fn

for tm in ['constant_bitrate', 'onoff', 'autocorrelated', 'modulated', 'all_multiplexed']:
    TEST_PATH = f'../../data/traffic_models/{tm}/test'
    labels_file = f'labels_delay_{tm}.npy'
    preds_file = f'predictions_delay_{tm}.npy'

    if os.path.exists(labels_file):
        labels = np.load(labels_file)
       # print(f"Loaded existing labels from {labels_file}")
    else:
        ds_test = input_fn(TEST_PATH, shuffle=False).take(200)
        labels = []
        for inputs, y in ds_test:
            labels.append(y.numpy())
        labels = np.concatenate(labels)
        np.save(labels_file, labels)
        print(f"Saved missing label file {labels_file}")

    if not os.path.exists(preds_file):
        print(f"Missing prediction file: {preds_file}. Skipping {tm}.")
        continue

    preds = np.load(preds_file)
    mape = np.abs(preds - labels) / labels * 100
    print(f"{tm:20s}  MAPE mean={mape.mean():.2f}%  median={np.median(mape):.2f}%  n={len(labels)}")