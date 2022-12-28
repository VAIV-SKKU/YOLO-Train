def Body(candle):
    return abs(candle.Open - candle.Close)


def Upper_Shadow(candle):
    return candle.High - max(candle.Open, candle.Close)


def Lower_Shadow(candle):
    return min(candle.Open, candle.Close) - candle.Low


def Candle(candle):
    return candle.High - candle.Low


def Bullish_Candle(candle):
    return candle.Open < candle.Close


def Bearish_Candle(candle):
    return candle.Open > candle.Close


def Bullish_Harami(date, section):
    dates = section.index.tolist()
    i = dates.index(date)
    
    if i < (len(dates)-1):
        c1 = section.loc[date]
        c2 = section.loc[dates[i+1]]
        condition = []
        condition.append(Bearish_Candle(c1))
        condition.append(Bullish_Candle(c2))
        condition.append(c1.Close < c2.Open)
        condition.append(c1.Open > c2.Close)
        condition.append(Body(c1) > 0.6*Candle(c1))
        return (False not in condition)
    return False


def Bullish_Engulfing(date, section):
    dates = section.index.tolist()
    i = dates.index(date)
    if i < (len(dates)-1):
        c1 = section.loc[date]
        c2 = section.loc[dates[i+1]]
        condition = []
        condition.append(Bearish_Candle(c1))
        condition.append(Bullish_Candle(c2))
        condition.append(c1.Close > c2.Open)
        condition.append(c1.Open < c2.Close)
        condition.append(Body(c2) > 0.6*Candle(c2))
        return (False not in condition)
    return False


def Bullish_Doji(date, section):
    dates = section.index.tolist()
    i = dates.index(date)
    if i < (len(dates)-1):
        c1 = section.loc[date]
        c2 = section.loc[dates[i+1]]
        condition = []
        condition.append(Bearish_Candle(c1))
        condition.append(c1.Low > c2.Low)
        condition.append(Upper_Shadow(c2) > 3*Body(c2))
        condition.append(Upper_Shadow(c2) > 3*Lower_Shadow(c2))
        condition.append(Body(c1) > 0.6*Candle(c1))
        return (False not in condition)
    return False


def Hammer(date, section):
    dates = section.index.tolist()
    i = dates.index(date)
    if i < (len(dates)-1):
        c1 = section.loc[date]
        c2 = section.loc[dates[i+1]]
        condition = []
        condition.append(Bearish_Candle(c1))
        condition.append(c1.Low > c2.Low)
        condition.append(Lower_Shadow(c2) > 2*Body(c2))
        condition.append(Upper_Shadow(c2) > 0.3*Body(c2))
        condition.append(Body(c1) > 0.6*Candle(c1))
        return (False not in condition)
    return False


def Morningstar(date, section):
    dates = section.index.tolist()
    i = dates.index(date)
    if (i < len(dates) - 2):
        c1 = section.loc[date]
        c2 = section.loc[dates[i+1]]
        c3 = section.loc[dates[i+2]]
        condition = []
        condition.append(Bearish_Candle(c1))
        condition.append(Bullish_Candle(c3))
        condition.append(Body(c1) > 0.6*Candle(c1))
        condition.append(c2.Open < c1.Close)
        condition.append(c3.Open > c2.Close)
        condition.append(Body(c2) < 0.3*Candle(c2))
        condition.append(Body(c2) < Body(c1))
        condition.append(Body(c2) < Body(c3))
        condition.append(c2.Low < c3.Low)
        condition.append(c2.Low < c1.Low)
        condition.append(c2.High < c1.Open)
        condition.append(c2.High < c3.Close)
        return (False not in condition)
    return False
    


def Bearish_Harami(date, section):
    dates = section.index.tolist()
    i = dates.index(date)  # 큰 음봉의 위치
    if i < (len(dates)-1):
        c1 = section.loc[date]
        c2 = section.loc[dates[i+1]]
        condition = []
        condition.append(Bullish_Candle(c1))
        condition.append(Bearish_Candle(c2))
        condition.append(c1.Close > c2.Open)
        condition.append(c1.Open < c2.Close)
        condition.append(Body(c1) > 0.6*Candle(c1))
        return (False not in condition)
    return False


def Bearish_Engulfing(date, section):
    dates = section.index.tolist()
    i = dates.index(date)  # 큰 양봉의 위치
    if i < (len(dates)-1):
        c1 = section.loc[date]
        c2 = section.loc[dates[i+1]]
        condition = []
        condition.append(Bullish_Candle(c1))
        condition.append(Bearish_Candle(c2))
        condition.append(c1.Close < c2.Open)
        condition.append(c1.Open > c2.Close)
        condition.append(Body(c2) > 0.6*Candle(c2))
        return (False not in condition)
    return False


def Gravestone_Doji(date, section):
    dates = section.index.tolist()
    i = dates.index(date)  # 큰 양봉의 위치
    if i < (len(dates) - 1):
        c1 = section.loc[date]
        c2 = section.loc[dates[i+1]]
        condition = []
        condition.append(Bullish_Candle(c1))
        condition.append(c1.High < c2.High)
        condition.append(Upper_Shadow(c2) > 3*Body(c2))
        condition.append(Upper_Shadow(c2) > 3*Lower_Shadow(c2))
        condition.append(Body(c1) > 0.6*Candle(c1))
        return (False not in condition)
    return False


def HangingMan(date, section):
    dates = section.index.tolist()
    i = dates.index(date)  # 큰 음봉의 위치
    if i < (len(dates) - 1):
        c1 = section.loc[date]
        c2 = section.loc[dates[i+1]]
        condition = []
        condition.append(Bullish_Candle(c1))
        condition.append(c1.High < c2.High)
        condition.append(Lower_Shadow(c2) > 2*Body(c2))
        condition.append(Upper_Shadow(c2) > 0.3*Body(c2))
        condition.append(Body(c1) > 0.6*Candle(c1))
        return (False not in condition)
    return False


def Eveningstar(date, section):
    dates = section.index.tolist()
    i = dates.index(date)
    if (i < len(dates) - 2):
        c1 = section.loc[date]
        c2 = section.loc[dates[i+1]]
        c3 = section.loc[dates[i+2]]
        condition = []
        condition.append(Bullish_Candle(c1))
        condition.append(Bearish_Candle(c3))
        condition.append(Body(c1) > 0.6*Candle(c1))
        condition.append(c2.Open > c1.Close)
        condition.append(c3.Open < c2.Close)
        condition.append(Body(c2) < 0.3*Candle(c2))
        condition.append(Body(c2) < Body(c1))
        condition.append(Body(c2) < Body(c3))
        condition.append(c2.High > c3.High)
        condition.append(c2.High > c1.High)
        condition.append(c2.Low > c1.Open)
        condition.append(c2.Low > c3.Close)
        return (False not in condition)
    return False


def Bullish(date, section):  # Buy
    pattern = ['Bullish Harami', 'Bullish Engulfing', 'Bullish Doji', 'Hammer', 'Morningstar']
    condition = []
    condition.append(Bullish_Harami(date, section))
    condition.append(Bullish_Engulfing(date, section))
    condition.append(Bullish_Doji(date, section))
    condition.append(Hammer(date, section))
    condition.append(Morningstar(date, section))
    result = {pattern[i]: condition[i] for i in range(5)}
    return result


def Bearish(date, section):  # Sell
    pattern = ['Bearish Harami', 'Bearish Engulfing', 'Gravestone Doji', 'Hanging Man', 'Eveningstar']
    condition = []
    condition.append(Bearish_Harami(date, section))
    condition.append(Bearish_Engulfing(date, section))
    condition.append(Gravestone_Doji(date, section))
    condition.append(HangingMan(date, section))
    condition.append(Eveningstar(date, section))
    result = {pattern[i]: condition[i] for i in range(5)}
    return result