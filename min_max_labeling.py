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

from manager import VAIV  # noqa: E402
from pattern import Bullish, Bearish  # noqa: E402


def get_point(vaiv: VAIV, section):
    ticker = vaiv.kwargs.get('ticker')
    pattern = vaiv.kwargs.get('pattern')
    trade_date = vaiv.kwargs.get('trade_date')
    section = section.astype(int)
    max_date = section.Close.idxmax()
    min_date = section.Close.idxmin()
    return min_date, max_date


def claculate_profit(left_close, right_close):
    return (right_close - left_close) / left_close * 100


def date_range(date, stock, left_thres, right_thres, temp):
    close = stock.Close.loc[date]
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


def define_priority(labeling):
    buy_labeling = labeling[labeling.index == 1].sort_values('Close')
    sell_labeling = labeling[labeling.index == 0].sort_values('Close', ascending=False)
    buy_length = len(buy_labeling)
    sell_length = len(sell_labeling)
    length = max(buy_length, sell_length)
    prioritys = []
    priority = 1
    while len(prioritys) < length:
        prioritys += pow(2, priority-1) * [priority]
        priority += 1
    prioritys = prioritys[:length]
    buy_labeling['Priority'] = prioritys[:buy_length]
    sell_labeling['Priority'] = prioritys[:sell_length]
    return pd.concat([buy_labeling, sell_labeling])


def make_min_max_span(vaiv: VAIV, span, priority):
    candle = vaiv.kwargs.get('candle')
    trade_date = vaiv.kwargs.get('trade_date')
    stock = vaiv.modedf.get('stock')

    total_dates = stock.index.tolist()
    trade_index = total_dates.index(trade_date)
    start = total_dates[trade_index-245]
    end = total_dates[trade_index-1]

    trade = stock.loc[start:end]
    dates = trade.index.tolist()

    label_list = []
    for i in range(0, candle, span):
        section = trade.iloc[i:i+span]
        min_date, max_date = get_point(vaiv, section)
        min_drange = date_range(min_date, trade, 4, 2, -1)
        max_drange = date_range(max_date, trade, 4, 2, 1)
        # buy: 1, sell: 0
        df = pd.DataFrame({
            'Label': [1, 0],
            'Date': [min_date, max_date],
            'Range': ['/'.join(min_drange), '/'.join(max_drange)],
            'Priority': [priority, priority],
        })
        label_list.append(df)
    return label_list


def make_min_max_labelings(vaiv: VAIV):
    ticker = vaiv.kwargs.get('ticker')
    trade_date = vaiv.kwargs.get('trade_date')
    candle = vaiv.kwargs.get('candle')
    span = candle
    priority = 1

    dividend = 2
    label_list = []
    while span > 10:
        span_label_list = make_min_max_span(vaiv, span, priority)
        label_list += span_label_list
        priority += 1
        span = candle // dividend
        dividend *= 2
        # dividend += 1
    labeling = pd.concat(label_list).drop_duplicates(subset=['Date']).set_index('Label')
    # priority_labeling = define_priority(labeling)
    vaiv.set_df('min_max', labeling)
    vaiv.save_df('min_max')


def make_ticker_labelings(vaiv: VAIV, start_date, end_date):
    predict = vaiv.modedf.get('predict')
    predict = predict.loc[start_date:end_date]
    for trade_date in predict.index.tolist():
        vaiv.set_kwargs(trade_date=trade_date)
        vaiv.load_df('min_max')
        make_min_max_labelings(vaiv)
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
        'pattern': False,
        'name': 'Kospi968M',  # Labeling folder 이름
    }
    vaiv.set_kwargs(**kwargs)
    vaiv.set_stock()
    vaiv.set_prediction()
    vaiv.set_image()
    vaiv.set_labeling()
    vaiv.make_dir(yolo=True, labeling=True)
    years = range(2006, 2023)  # 년도
    start_mds = ['01-01', '04-01', '07-01', '10-01']
    end_mds = ['03-31', '06-30', '09-30', '12-31']
    num = 968  # 종목 개수
    for year in years:
        year = f'{year}-'
        for start_md, end_md in zip(start_mds, end_mds):
            start_date = year + start_md
            end_date = year + end_md
            p = Process(target=make_all_labelings, args=(vaiv, start_date, end_date, num, ))
            p.start()
    # make_all_labelings(vaiv, start_date=start_date, end_date=end_date, num=50)
