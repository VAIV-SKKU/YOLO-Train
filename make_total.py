import pandas as pd
from pathlib import Path

def condition(dates: list, detect, trade_date, date_thres):
    try:
        i = dates.index(detect)
        j = dates.index(trade_date)
    except ValueError:
        return False
    if i - j < date_thres:
        return True
    else:
        return False
    
def main(date_thres):
    stock_path = '/home/ubuntu/2022_VAIV_Cho/VAIV/Common/Stock/Kospi'
    stock = Path(stock_path)
    
    roots = [
        Path('/home/ubuntu/2022_VAIV_Cho/VAIV/Yolo/Code/runs/detect/Kospi50M_0.8')
    ]
    
    for root in roots:
        signals = root / 'signals'
        total = root / 'total'
        total.mkdir(parents=True, exist_ok=True)
        signal = list(signals.iterdir())
        signal_list = []
        for fname in signal:
            ticker = fname.stem
            stock_df = stock_df = pd.read_csv(stock / fname.name, index_col='Date')
            dates = stock_df.index.tolist()
            
            f = pd.read_csv(fname)
            f['Ticker'] = ticker
            f = f[f.apply(lambda x: condition(dates, x['Detect'], x['Date'], date_thres), axis=1)].set_index('Date')
            signal_list.append(f)
        total_df = pd.concat(signal_list)
        total_df.to_csv(total / f'{root.stem}_{date_thres}.csv')
        
if __name__ == '__main__':
    main(2)