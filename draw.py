import argparse
import time
from pathlib import Path

import cv2
import torch
from tqdm import tqdm
import pandas as pd
import numpy as np
import sys

from models.experimental import attempt_load
from utils.datasets import LoadImages, letterbox
from utils.general import check_img_size, non_max_suppression, \
    apply_classifier, scale_coords, xyxy2xywh, xywh2xyxy, xywhn2xyxy, \
    set_logging, increment_path
# from utils.plots import plot_one_box
from utils.pixel import StockImage
from utils.torch_utils import select_device, load_classifier, \
    TracedModel
    
ROOT = Path('/home/ubuntu/2022_VAIV_Cho/VAIV')
sys.path.append(str(ROOT))
sys.path.append(str(ROOT / 'Common' / 'Code'))

from manager import VAIV  # noqa: E402


colors = [(63,63,219), (119,216,121)]  # bgr
colors = [(0, 0, 255), (0, 0, 255), (255, 0, 0), (255, 0, 0)]
labels = ['sell', 'buy']

def plot_one_box(x, img, mode=None, color=None, label=None, line_thickness=3):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        if mode == 'labeling':
            c1 = c1[0], c2[1]
            c2 = c1[0] + t_size[0], c1[1] + t_size[1] + 3
            cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
            cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
        else:
            c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
            cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
            cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
        
        
def get_xyxy(vaiv: VAIV, drange):
    pixel = vaiv.modedf.get('pixel')
    xmins = []
    ymins = []
    xmaxs = []
    ymaxs = []
    for date in drange:
        xmin, ymin, xmax, ymax = pixel.loc[date, 'Xmin':'Ymax'].tolist()
        xmins.append(xmin)
        ymins.append(ymin)
        xmaxs.append(xmax)
        ymaxs.append(ymax)
    return [min(xmins), min(ymins), max(xmaxs), max(ymaxs)]


def xyxy_to_xywh(xyxy):
    size = [1800, 650]
    shape = (size[1], size[0], 3)
    gn = torch.tensor(shape)[[1, 0, 1, 0]]
    xywh = (
        xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn
    ).view(-1).tolist()  # normalized xywh
    # xmin, ymin, xmax, ymax = xyxy
    # width, height = size
    # x = xmin / width
    # y = ymin / height
    # w = (xmax - xmin) / width
    # h = (ymax - ymin) / height
    # xywh = [x, y, w, h]
    return xywh

def draw(label, cls, im0, xyxy, mode=None):
    plot_one_box(
        xyxy, im0, mode, label=label,
        color=colors[int(cls)],
        line_thickness=2,
    )
    return im0


def detect_draw(vaiv: VAIV, im0):
    label_path = Path('/home/ubuntu/2022_VAIV_Cho/VAIV/Yolo/Code/runs/detect/000020_2021-01-06/labels')
    vaiv.set_fname('txt', ticker=ticker, trade_date=trade_date)
    vaiv.set_path(label_path)
    f = open(vaiv.path, 'r')
    
    while True:
        x = []
        line = f.readline().rstrip()
        if line == "":
            break
        label, *xywh, prob = line.split(' ')
        cls = int(label) + 2
        label = labels[int(label)] + f' {round(float(prob), 1)}'
        xywh = [float(pixel) for pixel in xywh]
        size = [1800, 650]
        shape = (size[1], size[0], 3)
        gn = torch.tensor(shape)[[1, 0, 1, 0]]
        xyxy = xywh2xyxy(torch.tensor(xywh).view(1,4) * gn).view(-1).tolist()
        im0 = draw(label, cls, im0, xyxy, 'detect')
    f.close()
    return im0
    
        
def label_draw(vaiv: VAIV, im0):
    vaiv.set_fname('png', ticker=ticker, trade_date=trade_date)
    vaiv.set_path(vaiv.common.image.get('images'))
    im0 = cv2.imread(str(vaiv.path))
    
    label_path = Path('/home/ubuntu/2022_VAIV_Cho/VAIV/Yolo/Dataset/MinMax50/train/labels')
    vaiv.set_fname('txt', ticker=ticker, trade_date=trade_date)
    vaiv.set_path(label_path)
    f = open(vaiv.path, 'r')
    
    while True:
        x = []
        line = f.readline().rstrip()
        if line == "":
            break
        label, *xywh = line.split(' ')
        cls = int(label)
        label = labels[int(label)]
        xywh = [float(pixel) for pixel in xywh]
        size = [1800, 650]
        shape = (size[1], size[0], 3)
        gn = torch.tensor(shape)[[1, 0, 1, 0]]
        xyxy = xywh2xyxy(torch.tensor(xywh).view(1,4) * gn).view(-1).tolist()
        im0 = draw(label, cls, im0, xyxy, 'labeling')
    f.close()
    return im0

    
def pattern_draw(vaiv:VAIV):
    vaiv.load_df('pattern')
    vaiv.load_df('pixel')
    ticker = vaiv.kwargs.get('ticker')
    trade_date = vaiv.kwargs.get('trade_date')
    patterns = vaiv.modedf.get('pattern').reset_index()
    pixels = vaiv.modedf.get('pixel')
    vaiv.set_fname('png', ticker=ticker, trade_date=trade_date)
    vaiv.set_path(vaiv.common.image.get('images'))
    im0 = cv2.imread(str(vaiv.path))
    duplicate = patterns[patterns.duplicated(subset=["Range"])]
    patterns.drop_duplicates(subset=["Range"], keep='first', inplace=True)
    
    for row in duplicate.to_dict('records'):
        date_range = row.get('Range')
        pattern = row.get('Pattern')
        i = patterns.index[patterns.Range == date_range][0]
        patterns.loc[i, 'Pattern'] = patterns.loc[i, 'Pattern'] + '/' + pattern
    
    patterns.reset_index()
    for row in patterns.to_dict('records'):
        label = row.get('Label')
        pattern = row.get('Pattern')
        date_range = row.get('Range').split('/')
        xyxy = get_xyxy(vaiv, date_range)
        im0 = draw(pattern, label, im0, xyxy)
    cv2.imwrite(vaiv.path.name, im0)    
    
    
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
        'name': 'MinMax50',
    }
    vaiv.set_kwargs(**kwargs)
    vaiv.set_stock()
    vaiv.set_prediction()
    vaiv.set_image()
    vaiv.set_labeling()
    vaiv.make_dir(yolo=True, labeling=True)
    
    ticker = "000157"
    trade_date = "2019-01-04"
    vaiv.set_kwargs(ticker=ticker, trade_date=trade_date)
    
    vaiv.set_fname('png', ticker=ticker, trade_date=trade_date)
    vaiv.set_path(vaiv.common.image.get('images'))
    im0 = cv2.imread(str(vaiv.path))
    
    im0 = label_draw(vaiv, im0)
    # im0 = detect_draw(vaiv, im0)
    cv2.imwrite(vaiv.path.stem + '.png', im0)
    quit()
    pattern_draw(vaiv)