import os
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import torch
import sys
import numpy as np
from multiprocessing import Process

os.environ['OPENBLAS_NUM_THREADS'] = '1'
sys.path.append('/home/ubuntu/2022_VAIV_Cho/VAIV/Yolo/Code/yolov7')
from utils.general import xyxy2xywh, xywh2xyxy  # noqa: E402

ROOT = Path('/home/ubuntu/2022_VAIV_Cho/VAIV')
sys.path.append(str(ROOT))
sys.path.append(str(ROOT / 'Common' / 'Code'))

from manager import VAIV  # noqa: E402
from pattern_labeling import make_pattern_labelings  # noqa: E402
from min_max_labeling import make_min_max_labelings  # noqa: E402
from merge_labeling import make_merge_labelings  # noqa: E402
from candlestick import make_candlestick  # noqa: E402


def xyxy_to_xywh(vaiv: VAIV, xyxy):
    size = vaiv.kwargs.get('size')
    shape = (size[1], size[0], 3)
    gn = torch.tensor(shape)[[1, 0, 1, 0]]
    xywh = (
        xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn
    ).view(-1).tolist()  # normalized xywh
    return np.round(xywh, 6)


def date_range(date, dates, num):
    now = dates.index(date)
    if now < num-1:
        return dates[0:num]
    elif now > (len(dates) - (num-1)):
        return dates[-num:len(dates)]
    else:
        plus = num // 2
        minus = num - plus
        return dates[now-minus:now+plus]


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


def make_ticker_labelings(vaiv: VAIV, start_date, end_date, offset):
    predict = vaiv.modedf.get('predict')
    stock = vaiv.modedf.get('stock')
    stock_dates = stock.index.tolist()
    if predict.empty:  # 245일 이상
        return
    predict = predict[(predict.index >= start_date) & (predict.index <= end_date)]
    dates = predict.index.tolist()
    for i in range(0, len(dates), offset):
        trade_date = dates[i]
        # print(trade_date)
        try:
            ti = stock_dates.index(trade_date)
        except ValueError:  # There is no such date in stock data
            continue
        try:
            start = stock_dates[ti-245]
        except IndexError:
            print('------------IndexError-----------')
            print(len(stock_dates), ti)
            print()
        end = stock_dates[ti-1]
        pred = pd.Series({'Start': start, 'End': end, 'Date': trade_date})
        vaiv.set_kwargs(trade_date=trade_date)
        make_candlestick(vaiv, stock, pred)
        vaiv.load_df('pixel')
        vaiv.load_df('min_max')
        # make_min_max_labelings(vaiv)
        vaiv.load_df('pattern')
        # make_pattern_labelings(vaiv)
        vaiv.load_df('merge')
        # try:
        make_merge_labelings(vaiv, 4, 2)
        # except KeyError as e:
        #     print('KeyError: ', vaiv.kwargs.get('ticker'), trade_date)
        #     exit(1)
        #     continue
    return


def make_all_labelings(vaiv: VAIV, start_date='2006', end_date='z', num=968, offset=10):
    market = vaiv.kwargs.get('market')
    vaiv.load_df(market)
    df = vaiv.modedf.get(market).reset_index()

    # pbar = tqdm(total=num)
    for ticker in df.Ticker.tolist()[:num]:
        vaiv.set_kwargs(ticker=ticker)
        vaiv.load_df('stock')
        vaiv.load_df('predict')
        make_ticker_labelings(vaiv, start_date, end_date, offset)
        # pbar.update()
    # pbar.close()


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
        'name': 'Kospi942_2006-2022_10',  # Labeling 폴더 이름 (기존과 겹치지 않게 정해야 한다)
    }
    vaiv.set_kwargs(**kwargs)
    vaiv.set_stock()
    vaiv.set_prediction()
    vaiv.set_image()
    vaiv.set_labeling()
    vaiv.make_dir(yolo=True, labeling=True)
    years = range(2006, 2023)  # 년도 (2006 <= year < 2023)
    start_mds = ['01-01', '04-01', '07-01', '10-01']
    end_mds = ['03-31', '06-30', '09-30', '12-31']
    num = 942  # 종목 개수
    offset = 10
    for year in years:
        year = f'{year}-'
        for start_md, end_md in zip(start_mds, end_mds):
            start_date = year + start_md
            end_date = year + end_md
            p = Process(target=make_all_labelings, args=(vaiv, start_date, end_date, num, offset, ))
            p.start()
    # make_all_labelings(vaiv, start_date=start_date, end_date=end_date, num=50)
