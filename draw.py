import argparse
import cv2
import random
from pathlib import Path
import sys

p = Path.absolute(Path.cwd().parent)
sys.path.append(str(p))
from Data.candlestick import YoloChart
from Data.labeling import YoloLabeling


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


def get_xyxy(pixel, drange):
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


def draw(label, cls, img, xyxy, mode=None):
    colors = [(63, 63, 219), (119, 216, 121)]  # bgr
    plot_one_box(
        xyxy, img, mode, label=label,
        color=colors[int(cls)],
        line_thickness=2,
    )
    return img



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--market', '-m', type=str, dest='market', required=True,
        help='You can input a market under options\n' + \
             'KOSPI: Stock market includes KOSPI only\n' + \
             'KOSDAQ: Stock market includes KOSDAQ only\n' + \
             'KONEX: Stock market includes KONEX only'
    )
    parser.add_argument(
        '--method', type=str, required=True, help='the method of labeling'
    )
    parser.add_argument(
        '--ticker', '-t', type=str, required=True, help='ticker to draw'
    )
    parser.add_argument(
        '--date', '-d', type=str, required=True, help='trade date to draw'
    )
    
    args = parser.parse_args()
    chart = YoloChart(market=args.market, exist_ok=True)
    labeling = YoloLabeling(market=args.market, method=args.method)
    
    
    pixel = chart.load_pixel_coordinates(ticker=args.ticker, trade_date=args.date)
    img_path = chart.load_chart_path(ticker=args.ticker, trade_date=args.date)
    img = cv2.imread(str(img_path))
    l = labeling.load_labeling(ticker=args.ticker, trade_date=args.date)
    
    if args.method == 'MinMax':
        label_dict = {0: 'sell', 1: 'buy'}
        for row in l.to_dict('records'):
            cls = row['Label']
            label = label_dict[int(cls)]
            drange = [row['Date']]
            xyxy = get_xyxy(pixel, drange)
            img = draw(label, row['Label'], img, xyxy)
    
    elif args.method == 'Pattern':
        patterns = dict()
        label_dict = dict()
        for row in l.to_dict('records'):
            cls = row['Label']
            label = row['Pattern']
            label_dict[int(cls)] = label
            drange = row['Range']
            if drange in patterns:
                patterns[drange].append(cls)
            else:
                patterns[drange] = [cls]
        
        for drange, pattern in patterns.items():
            drange = drange.split('/')
            label = ','.join([label_dict[int(cls)] for cls in pattern])
            cls = pattern[0] // 5
            xyxy = get_xyxy(pixel, drange)
            img = draw(label, cls, img, xyxy)
    
    elif args.method == 'Merge':
        label_dict = {0: 'sell', 1: 'buy'}
        for row in l.to_dict('records'):
            cls = row['Label']
            label = label_dict[int(cls)]
            drange = row['Range'].split('/')
            xyxy = get_xyxy(pixel, drange)
            img = draw(label, row['Label'], img, xyxy)

    cv2.imwrite(img_path.name, img)