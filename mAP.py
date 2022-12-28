import pandas as pd
import numpy as np
from pathlib import Path
import argparse


def SignalCount(dir):
    p = Path(dir)
    files_path = list(p.iterdir())

    total = 0
    labels = []
    for file_path in files_path:
        f = open(file_path, 'r')
        lines = list(enumerate(f))
        signals = [s[1] for s in lines]
        labels += [int(s.split(' ')[0]) for s in signals]
        total += len(lines)
    return labels


def DateCount(dir):
    p = Path(dir).parent / 'dataframes'
    files_path = list(p.iterdir())

    drange = 0
    num = 0
    for file_path in files_path:
        df = pd.read_csv(file_path)
        dates = df.Range.apply(lambda d: len(list(filter(None, d.split('/')))))
        drange += dates.sum()
        num += len(dates)
    return drange / num


def rangeCount(df, col):
    rangeDict = {}
    for i in range(10):
        rangei = df[(df[col] >= i/10)]
        rangeDict[i+1] = round(len(rangei) / len(df) * 100, 1)
        print(f'\t{col} {(i+1)/10}: {round(len(rangei) / len(df) * 100, 1)}%')
        # meanCount(rangei, 'DateD')
    # for k, v in rangeDict.items():
    #     print(f'{col} {k/10}: {v}%')
    # range1 = df[df.col < 0.1]
    # range2 = df[(df.col < 0.2) & (df.col >= 0.1)]


def meanCount(df, col):
    df = df[df.IoU != 0]
    colList = df[col].tolist()
    if not colList:
        return
    print(f'\t\t{col} Min: ', min(colList))
    print(f'\t\t{col} Max: ', max(colList))
    print(f'\t\t{col} Average: ', round(np.mean(colList), 2))


def mAPbyClass(df, signal, label=None):
    if label is None:
        print('Average mAP')
        signal = len(signal)
    else:
        print(f'Label: {label}')
        df = df[df.Label == label]
        signal = signal.count(label)

    for iou_thres in np.arange(opt.iou_thres, 1, 0.1):
        print('Iou threshold: {:.1f}'.format(iou_thres))
        TPdf = df[(df['IoU'] > iou_thres)].reset_index(drop=True)
        TP = len(TPdf)
        # FP = len(df) - TP
        # FN = signal - TP
        precision = TP / (len(df))
        recall = TP / (signal)
        # print(TP, signal)
        print(len(TPdf[TPdf['DateD'] == 0])/TP)
        print(f'\tPrecision: {round(precision,4)} ({TP}/{len(df)})\n\tRecall: {round(recall,4)} ({TP}/{signal})')
    # rangeCount(df, 'Probability')
    # rangeCount(df, 'IoU')
    print()
    # meanCount(score, 'DateD')
    # rangeCount(TPdf, 'Probability')
    # meanCount(TPdf, 'DateD')
    print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-d', '--directory', type=str, help='Directory of text files'
    )
    parser.add_argument(
        '-s', '--score', type=str, help='Path of score file'
    )
    parser.add_argument(
        '-iou', '--iou-thres', default=0.6, type=float, help='iou thres'
    )
    parser.add_argument(
        '-nc', '--nc', default=0.6, type=int, help='number of class'
    )
    opt = parser.parse_args()

    score = pd.read_csv(opt.score)
    signal = SignalCount(opt.directory)

    print('Date Range: {0}일'.format(DateCount(opt.directory)))
    mAPbyClass(score, signal)
    for nc in range(opt.nc):
        try:
            mAPbyClass(score, signal, nc)
        except ZeroDivisionError:
            continue
