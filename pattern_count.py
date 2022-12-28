from pathlib import Path
from tqdm import tqdm
import pandas as pd
import sys
import numpy as np
import multiprocessing as mp

ROOT = Path('/home/ubuntu/2022_VAIV_Cho/VAIV')
sys.path.append(str(ROOT))
sys.path.append(str(ROOT / 'Common' / 'Code'))

from manager import VAIV  # noqa: E402


def count_pattern(vaiv:VAIV):
    p = vaiv.yolo.labeling.get('merge')
    filesPath = list(p.iterdir())
    pattern_count = {'Bullish Harami': 0, 'Bullish Engulfing': 0, 'Bullish Doji': 0, 'Hammer': 0, 'Morningstar': 0, 'Bearish Harami': 0, 'Bearish Engulfing': 0, 'Gravestone Doji': 0, 'Hanging Man': 0, 'Eveningstar': 0}
    priority_count = {}
    pattern_total = 0
    priority_total = 0
    for i, filePath in enumerate(tqdm(filesPath)):
        labeling = pd.read_csv(filePath)
        labeling.fillna('', inplace=True)
        for row in labeling.to_dict('records'):
            patterns = list(set(row.get('Pattern').split('/')))
            patterns = list(filter(None, patterns))
            priority = row.get('Priority')
            priority_count.setdefault(priority, [0, 0])
            priority_count[priority][1] += 1
            if not patterns:
                continue
            for pattern in patterns:
                pattern_total += 1
                pattern_count[pattern] += 1
            priority_count[priority][0] += 1
            priority_total += 1
            
    # pattern_count = {k: round(v / pattern_total * 100, 2) for k, v in pattern_count.items()}
    # priority_count = {k: round(v[0] / v[1] * 100, 1) for k, v in priority_count.items()}
    # priority_count = {k: round(v[0] / priority_total * 100, 1) for k, v in priority_count.items()}
    # priority_count = {k: v[0] for k, v in priority_count.items()}
    priority_count = dict(sorted(priority_count.items()))
    priority_count = {k: [round(v[0] / len(filesPath), 1), round(v[1] / len(filesPath), 1)] for k, v in priority_count.items()}
    
    for pattern, count in pattern_count.items():
        print(f'{pattern}: {count}')
    for prior, count in priority_count.items():
        print(f'Priority{prior}: {count[0]} {count[1]}')
    
if __name__ == '__main__':
    vaiv = VAIV(ROOT)
    kwargs = {
        'market': 'Kospi',
        'feature': {'Volume': False, 'MA': [-1], 'MACD': False},
        'offset': 1,
        'size': [1800, 650],
        'candle': 245,
        'linespace': 1,
        'candlewidth': 0.8,
        'style': 'default',
        'folder': 'yolo',
        'name': 'Kospi50D_20',
    }
    vaiv.set_kwargs(**kwargs)
    vaiv.set_stock()
    vaiv.set_prediction()
    vaiv.set_image()
    vaiv.set_labeling()
    vaiv.make_dir(yolo=True, labeling=True)
    
    count_pattern(vaiv)