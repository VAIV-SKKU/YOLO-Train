import pandas as pd
import numpy as np
from pathlib import Path
import exchange_calendars as xcals
xkrx = xcals.get_calendar("XKRX")


def condition(dates: list, detect, trade_date, date_thres):
    i = dates.index(detect)
    j = dates.index(trade_date)
    # if i - j == 1:
    #     print(trade_date, detect)
    #     quit()
    if i - j < date_thres:
        return True
    else:
        return False


count = {}
count_date = {}
p = '/home/ubuntu/2022_VAIV_Cho/VAIV/Yolo/Code/runs/detect/Kospi50M/signals'
signal = Path(p)
stock_path = '/home/ubuntu/2022_VAIV_Cho/VAIV/Common/Stock/Kospi'
stock = Path(stock_path)
f = list(signal.iterdir())


def trade_count(year, date_thres):
    trade = {}
    for file in f:
        df = pd.read_csv(file)
        con = (df.Date > f'{year}') & (df.Date < f'{year+1}')
        df = df[con]
        ticker = file.stem
        stock_df = stock_df = pd.read_csv(stock / file.name, index_col='Date')
        dates = stock_df.index.tolist()
        trade[ticker] = {
            'Buy': [],
            'Sell': [],
            'Buy_Price': [],
            'Sell_Price': [],
            'Profit': [],
        }
        for dic in df.to_dict('records'):
            trade_date = dic.get('Date')
            label = dic.get('Label')
            detect = dic.get('Detect')
            buy_num = len(trade[ticker].get('Buy'))
            sell_num = len(trade[ticker].get('Sell'))

            if (detect not in dates) or (trade_date not in dates):
                continue

            if condition(dates, detect, trade_date, date_thres):
                if (label == 'buy') & (buy_num == sell_num):
                    trade[ticker]['Buy'].append(trade_date)
                    trade[ticker]['Buy_Price'].append(stock_df.loc[trade_date, 'Close'])
                    buy_price = stock_df.loc[trade_date, 'Close']
                if (label == 'sell') & (buy_num > sell_num):
                    sell_price = stock_df.loc[trade_date, 'Close']
                    profit = ((sell_price - buy_price) / buy_price)*100
                    # if profit > 0:
                    trade[ticker]['Sell'].append(trade_date)
                    trade[ticker]['Sell_Price'].append(stock_df.loc[trade_date, 'Close'])
                    trade[ticker]['Profit'].append(profit)
        if len(trade[ticker].get('Buy')) > len(trade[ticker].get('Sell')):
            trade[ticker]['Buy'].pop()
            trade[ticker]['Buy_Price'].pop()
    return trade


def buy_sell_count():
    for file in f:
        df = pd.read_csv(file)
        ticker = file.stem
        stock_df = pd.read_csv(stock / file.name)
        dates = stock_df['Date'].tolist()

        count[ticker] = {
            'Buy': 0,
            'Sell': 0,
            'Last_Buy': 0,
            'Last_Sell': 0,
        }
        for dic in df.to_dict('records'):
            trade_date = dic.get('Date')
            if trade_date not in count_date:
                count_date[trade_date] = {'Buy': 0, 'Sell': 0}
            if (dic.get('Detect') not in dates) or (trade_date not in dates):
                continue
            if dic.get('Label') == 'BUY':
                count[ticker]['Buy'] += 1
                if condition(dates, dic.get('Detect'), trade_date, 3):
                    count_date[trade_date]['Buy'] += 1
                    count[ticker]['Last_Buy'] += 1
            else:
                count[ticker]['Sell'] += 1
                if condition(dates, dic.get('Detect'), trade_date, 3):
                    count_date[trade_date]['Sell'] += 1
                    count[ticker]['Last_Sell'] += 1

# df = pd.DataFrame.from_dict(count, orient='index')
# df.index.name = 'Ticker'
# df.replace(0, np.NaN, inplace=True)
# df.dropna(inplace=True)
# total = pd.DataFrame({
#         'Ticker': ['Total', 'Mean'],
#         'Buy': [
#             sum(df['Buy']),
#             round(sum(df['Buy']) / len(df), 2)
#         ],
#         'Sell': [
#             sum(df['Sell']),
#             round(sum(df['Sell']) / len(df), 2)
#         ],
#         'Last_Buy': [
#             sum(df['Last_Buy']),
#             round(sum(df['Last_Buy']) / len(df), 2)
#         ],
#         'Last_Sell': [
#             sum(df['Last_Sell']),
#             round(sum(df['Last_Sell']) / len(df), 2)
#         ]
#     })
# total.set_index('Ticker', inplace=True)
# df = pd.concat([df, total])


def trade_result(trade, year):
    total = 0
    total_profit = 0
    total_buy = 0
    total_sell = 0
    dates = xkrx.sessions_in_range(f'{year}-01-01', f'{year}-12-31')
    dates = dates.strftime("%Y-%m-%d").tolist()
    date_profit = {date: 0 for date in dates}
    fore = []
    minus = 0
    plus = 0
    for result in trade.values():
        profit = result.get('Profit')
        buy_date = result.get('Buy')
        sell_date = result.get('Sell')
        for i in range(len(buy_date)):
            date_profit[buy_date[i]] += profit[i]
            fore.append(dates.index(sell_date[i]) - dates.index(buy_date[i]))
            # profit[i] = profit[i] / forecast
        minus_profit = [p for p in profit if p < 0]
        plus_profit = [p for p in profit if p > 0]
        minus += len(minus_profit)
        plus += len(plus_profit)
        total += len(profit)
        total_profit += sum(profit)
        total_buy += sum(result.get('Buy_Price'))
        total_sell += sum(result.get('Sell_Price'))
    print(plus, minus)
    try:
        # total_profit = (total_sell - total_buy) / total_buy * 100  # profit2 오버라이드
        avg = total_profit / total
        # avg = total_profit
        fore_avg = sum(fore) / len(fore)
    except ZeroDivisionError:
        total_profit = 0
        avg = 0
        fore_avg = 0
    for i in range(len(dates)-1):
        previous = dates[i]
        date_profit[dates[i+1]] += date_profit.get(previous)
    # print(date_profit)
    # quit()
    # date_avg = total_profit / len(dates)
    trade_avg = total / len(dates)
    return total, round(trade_avg, 1), round(avg, 1), round(fore_avg, 1), date_profit
    # print(f'거래량: {total}\n   ', end='')
    # print(f'일 평균 거래량: {trade_avg}\n   ', end='')
    # print(f'수익률: {round(total_profit, 1)}%\n   ', end='')
    # print(f'거래 평균 수익률: {round(avg, 1)}%\n   ', end='')
    # print(f'일 평균 수익률: {round(date_avg, 1)}')


def print_date(count_date, year):
    df_date = pd.DataFrame.from_dict(count_date, orient='index')
    df_date.index.name = 'Date'
    condition = (df_date.index > f'{year}') & (df_date.index < f'{year+1}')
    df_date = df_date.loc[condition]
    df_date.replace(0, np.NaN, inplace=True)
    df_date.dropna(inplace=True)
    total = pd.DataFrame({
            'Date': ['Total', 'Mean'],
            'Buy': [
                sum(df_date['Buy']),
                round(sum(df_date['Buy']) / len(df_date), 2)
            ],
            'Sell': [
                sum(df_date['Sell']),
                round(sum(df_date['Sell']) / len(df_date), 2)
            ]
        })
    total.set_index('Date', inplace=True)
    df_date = pd.concat([df_date, total])
    print(df_date)


# buy_sell_count()
# print_date(count_date, 2019)
# print_date(count_date, 2020)
# print_date(count_date, 2021)

def save_result(date_tlist, year_list):
    results = []
    for date_thres in date_tlist:
        print(f'Date_thres: {date_thres}')
        for year in year_list:
            trade = trade_count(year, date_thres)
            total, trade_avg, avg, fore, date_profit = trade_result(trade, year)
            result = pd.DataFrame({
                'Date_thres': [date_thres],
                '연도': [year],
                '거래량': [total],
                '일평균거래량': [trade_avg],
                '평균거래기간': [fore],
                '평균수익률': [avg]
            })
            result.set_index(['Date_thres'], inplace=True, drop=True)
            results.append(result)
    results = pd.concat(results)
    print(results)
    # results.to_csv('./Profit/Turkish_0.2_profit.csv')


save_result([1], [2021])
# profit = pd.read_csv('profit.csv', index_col=0)
# profit['평균거래기간'] = profit['평균거래기간'].round(1)
# print(profit)
# del profit['Unnamed: 0']
# profit.set_index('Date_thres', inplace=True)
# profit.to_csv('profit.csv')
