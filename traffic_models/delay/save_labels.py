import sys
sys.path.append('.')
sys.path.append('../..')
import numpy as np
from data_generator import input_fn

for tm in ['constant_bitrate', 'onoff', 'autocorrelated', 'modulated', 'all_multiplexed']:
    TEST_PATH = f'../../data/traffic_models/{tm}/test'
    ds_test = input_fn(TEST_PATH, shuffle=False).take(200)

    labels = []
    for inputs, y in ds_test:
        labels.append(y.numpy())
    labels = np.concatenate(labels)
    np.save(f'labels_delay_{tm}.npy', labels)

    preds = np.load(f'predictions_delay_{tm}.npy')
    mape = np.abs(preds - labels) / labels * 100
    print(f"{tm:20s}  MAPE mean={mape.mean():.2f}%  median={np.median(mape):.2f}%  n={len(labels)}")