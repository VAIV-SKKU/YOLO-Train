from pathlib import Path
from tqdm import tqdm
import pandas as pd
import sys
from multiprocessing import Process
sys.path.append('/home/ubuntu/2022_VAIV_Cho/VAIV/Yolo/Code/yolov7')

ROOT = Path('/home/ubuntu/2022_VAIV_Cho/VAIV')
sys.path.append(str(ROOT))
sys.path.append(str(ROOT / 'Common' / 'Code'))

from manager import VAIV  # noqa: E402
from pattern import Bullish, Bearish  # noqa: E402


date_range = {'Bullish Harami': 2, 'Bullish Engulfing': 2, 'Bullish Doji': 2, 'Hammer': 2, 'Morningstar': 3, 'Bearish Harami': 2, 'Bearish Engulfing': 2, 'Gravestone Doji': 2, 'Hanging Man': 2, 'Eveningstar': 3}


def make_pattern_labelings(vaiv: VAIV):
    ticker = vaiv.kwargs.get('ticker')
    trade_date = vaiv.kwargs.get('trade_date')
    stock = vaiv.modedf.get('stock')
    predict = vaiv.modedf.get('predict')
    start = predict.loc[trade_date, 'Start']
    end = predict.loc[trade_date, 'End']
    stock = stock.loc[start:end]
    dates = stock.index.tolist()
    buy_count = 0
    sell_count = 0
    pattern_list = []
    for i, date in enumerate(dates):
        bullish = Bullish(date, stock)
        bearish = Bearish(date, stock)
        for pattern, check in bullish.items():
            label = list(date_range.keys()).index(pattern)
            if check:
                buy_count += 1
                df = pd.DataFrame({'Label': [label], 'Range': '/'.join(dates[i:i+date_range[pattern]]), 'Pattern': pattern})
                pattern_list.append(df)
        for pattern, check in bearish.items():
            label = list(date_range.keys()).index(pattern)
            if check:
                sell_count += 1
                df = pd.DataFrame({'Label': [label], 'Range': '/'.join(dates[i:i+date_range[pattern]]), 'Pattern': pattern})
                pattern_list.append(df)

    patterns = pd.concat(pattern_list).set_index('Label')
    vaiv.set_df('pattern', patterns)
    vaiv.save_df('pattern')


def make_ticker_labelings(vaiv: VAIV, start_date, end_date):
    predict = vaiv.modedf.get('predict')
    predict = predict.loc[start_date:end_date]
    for trade_date in predict.index.tolist():
        vaiv.set_kwargs(trade_date=trade_date)
        vaiv.load_df('pattern')
        make_pattern_labelings(vaiv)
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
        'name': 'Kospi968M',  # Labeling 폴더 이름
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

            quit()
    # make_all_labelings(vaiv, start_date=start_date, end_date=end_date, num=50)
