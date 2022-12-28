from pathlib import Path
from tqdm import tqdm
import pandas as pd
import torch
import sys
import numpy as np
from multiprocessing import Process
sys.path.append('/home/ubuntu/2022_VAIV_Cho/VAIV/Yolo/Code/yolov7')
from utils.general import xyxy2xywh, xywh2xyxy  # noqa: E402

ROOT = Path('/home/ubuntu/2022_VAIV_Cho/VAIV')
sys.path.append(str(ROOT))
sys.path.append(str(ROOT / 'Common' / 'Code'))
MAX_DATE = 5

from manager import VAIV  # noqa: E402
from pattern_labeling import make_pattern_labelings  # noqa: E402
from min_max_labeling import make_min_max_labelings  # noqa: E402


def xyxy_to_xywh(vaiv: VAIV, xyxy):
    size = vaiv.kwargs.get('size')
    shape = (size[1], size[0], 3)
    gn = torch.tensor(shape)[[1, 0, 1, 0]]
    xywh = (
        xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn
    ).view(-1).tolist()  # normalized xywh
    return np.round(xywh, 6)


def claculate_profit(left_close, right_close):
    return (right_close - left_close) / left_close * 100


def date_range(date, stock, left_thres, right_thres, temp):
    try:
        close = stock.Close.loc[date]
    except KeyError:
        print('Date: ', date)
        print(stock)
        exit(1)
    left_close = stock.Close.loc[:date].iloc[:-1].iloc[::-1]
    right_close = stock.Close.loc[date:].iloc[1:]
    left_most = date
    right_most = date

    for left, lclose in left_close.items():
        profit = temp*claculate_profit(lclose, close)
        if profit >= left_thres:
            left_most = left
            break
        if profit < 0:
            break

    for right, rclose in right_close.items():
        profit = -temp*claculate_profit(close, rclose)
        if profit >= right_thres:
            right_most = right
            break
        if profit < 0:
            break

    return stock.loc[left_most:right_most].index.tolist()


def get_xyxy(vaiv: VAIV, drange):
    pixel = vaiv.modedf.get('pixel')
    xmins = []
    ymins = []
    xmaxs = []
    ymaxs = []
    for date in drange:
        try:
            xmin, ymin, xmax, ymax = pixel.loc[date, 'Xmin':'Ymax'].tolist()
        except KeyError:
            print('------------KeyError-----------')
            print('Ticker: ', vaiv.kwargs.get('ticker'))
            print('Trade Date: ', vaiv.kwargs.get('trade_date'))
            print('Find: ', date)
            print(pixel)
            print()
            return
        xmins.append(xmin)
        ymins.append(ymin)
        xmaxs.append(xmax)
        ymaxs.append(ymax)
    return [min(xmins), min(ymins), max(xmaxs), max(ymaxs)]


def make_merge_labelings(vaiv: VAIV, left_thres, right_thres):
    ticker = vaiv.kwargs.get('ticker')
    trade_date = vaiv.kwargs.get('trade_date')
    min_max = vaiv.modedf.get('min_max').sort_values(by='Date')
    patterns = vaiv.modedf.get('pattern')
    stock = vaiv.modedf.get('stock')
    predict = vaiv.modedf.get('predict')
    start = predict.Start.loc[trade_date]
    end = predict.End.loc[trade_date]
    stock = stock.loc[start:end]
    # dates = stock.index.tolist()

    label_list = []
    before_drange = {'Label': -1, 'Date': 'empty', 'Range': []}
    if len(stock) < vaiv.kwargs.get('candle'):
        print('Predict Error: ', ticker, trade_date)
        return
    for row in min_max.reset_index().to_dict('records'):
        label = row.get('Label')
        date = row.get('Date')
        priority = row.get('Priority')

        if label == 1:
            temp = -1
        else:
            temp = 1

        # if (priority > prior_thres):
        #     continue
        drange = date_range(date, stock, left_thres, right_thres, temp)
        if len(drange) < 3:
            continue
        # print(date, drange)
        pattern = []
        # if (priority > pattern_thres):
        for row in patterns[(patterns.index // 5) == (1 - label)].to_dict('records'):
            pattern_drange = row.get('Range').split('/')
            intersection = list(set(drange).intersection(pattern_drange))
            if len(intersection) > 0:
                drange = list(set(drange).union(pattern_drange))
                pattern.append(row.get('Pattern'))
        pattern = list(set(pattern))
            # if len(pattern) == 0:
            #     continue

        # remove local minimum
        if before_drange.get('Label') == label:
            # intersection = list(set(drange).intersection(before_drange.get('Range')))
            # if len(intersection) > 0:
            close = temp*stock.Close.loc[date]
            before_close = temp*stock.Close.loc[before_drange.get('Date')]
            if close < before_close:
                continue
            else:
                label_list.pop()

        xyxy = get_xyxy(vaiv, drange)
        xywh = xyxy_to_xywh(vaiv, xyxy)

        df = pd.DataFrame({
            'Label': [label],
            'CenterX': [xywh[0]],
            'CenterY': [xywh[1]],
            'Width': [xywh[2]],
            'Height': [xywh[3]],
            'Range': ['/'.join(sorted(drange))],
            'Pattern': ['/'.join(pattern)],
            'Priority': [priority],
        })
        label_list.append(df)
        # print(drange)
        before_drange = {'Label': label, 'Date': date, 'Range': drange}
    try:
        labeling = pd.concat(label_list).set_index('Label')
    except ValueError:  # minmax나 pattern이 없다
        print('ValueError: ', ticker, trade_date, len(min_max), len(pattern))
        return
    vaiv.set_df('merge', labeling)
    vaiv.save_df('merge')


def make_ticker_labelings(vaiv: VAIV, start_date, end_date):
    predict = vaiv.modedf.get('predict')
    predict = predict.loc[start_date:end_date]
    for trade_date in predict.index.tolist():
        vaiv.set_kwargs(trade_date=trade_date)
        vaiv.load_df('pixel')
        vaiv.load_df('min_max')
        vaiv.load_df('pattern')
        vaiv.load_df('merge')
        make_merge_labelings(vaiv, 4, 2)
    return


def make_all_labelings(vaiv: VAIV, start_date='2006', end_date='z', num=968):
    market = vaiv.kwargs.get('market')
    vaiv.load_df(market)
    df = vaiv.modedf.get(market).reset_index()

    pbar = tqdm(total=num)
    for ticker in df.Ticker.tolist()[:num]:
        vaiv.set_kwargs(ticker=ticker)
        vaiv.load_df('stock')
        vaiv.load_df('predict')
        make_ticker_labelings(vaiv, start_date, end_date)
        pbar.update()
    pbar.close()


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
        'name': 'Kospi5968M',
    }
    vaiv.set_kwargs(**kwargs)
    vaiv.set_stock()
    vaiv.set_prediction()
    vaiv.set_image()
    vaiv.set_labeling()
    vaiv.make_dir(yolo=True, labeling=True)
    years = ['2019-', '2020-', '2021-']
    start_mds = ['01-01', '04-01', '07-01', '10-01']
    end_mds = ['03-31', '06-30', '09-30', '12-31']
    num = 50
    for year in years:
        for start_md, end_md in zip(start_mds, end_mds):
            start_date = year + start_md
            end_date = year + end_md
            p = Process(target=make_all_labelings, args=(vaiv, start_date, end_date, num, ))
            p.start()
