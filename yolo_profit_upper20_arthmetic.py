import pandas as pd
import exchange_calendars as xcals
import numpy as np
import matplotlib.pyplot as plt
from mplfinance.original_flavor import candlestick2_ochl, volume_overlay
from matplotlib import pyplot as plt, dates as mdates
from datetime import datetime as dt
import exchange_calendars as xcals 

def yoloboom(): 
    old_df  = pd.read_csv("/home/ubuntu/2022_VAIV_HyunJoon/yolov5/runs/detect/test_total_2021/signals/2021_KOSPI_YOLO_trading.csv",index_col=0) 
    #  마지막 날  buy-sell pair csv 생성하는 코드 
    old_df.reset_index(inplace=True, drop=True)

    buy_date = None
    ticker= None
    sell_date = None
    profit_sign = None
    profit_co=0
    profit_only=0
    profit_dict=[]

    plus_count=0
    minus_count=0


    # 시장 개장일들만 리스트로 변환하여 저장.
    XKRX = xcals.get_calendar("XKRX")
    pred_Dates = XKRX.sessions_in_range("2021-01-01","2021-12-31")
    pred_Dates = pred_Dates.strftime("%Y-%m-%d").tolist()

    # 1 pair의 buy-sell로 pair 수익률 구하기.
    for trading_index in old_df.index:
        if old_df['Label'][trading_index]==0: # sell 신호(0) 이면
            ticker = old_df['Ticker'][trading_index]
            buy_index = trading_index-1 # buy 신호(1) 위치.
            buy_date = old_df['Date'][buy_index]
            sell_date = old_df['Date'][trading_index]
            buy_price = old_df['Close'][buy_index]
            sell_price = old_df['Close'][trading_index]
            buy_probability = old_df['Probability'][buy_index]
            sell_probability = old_df['Probability'][trading_index]
        
            profit_co =  round( ( sell_price*0.9975 - buy_price ) / buy_price * 100 , 3) 
            profit_only =  round( ( sell_price - buy_price ) / buy_price * 100, 3 )

            #+ 수익률
            if profit_co>0:
                plus_count+=1
                profit_sign = '+'
  
            #- 수익률
            elif profit_co<0:
                minus_count+=1
                profit_sign = '-'
  
            profit_row= [ ticker, buy_date, buy_price, buy_probability,  sell_date, sell_price, sell_probability, profit_only, profit_co, profit_sign ]
            profit_dict.append(profit_row)


    #2 buy-sell pair dataframe 생성
    profit_pair_df = pd.DataFrame(profit_dict,columns=['Ticker','Buy_Date','Buy_Price','Buy_Prob', 'Sell_Date','Sell_Price', 'Sell_Prob', 'Pair_순수익률', 'Pair_수수료_수익률', '수익률_부호'])
    #3 dataframe을 csv로 생성
    profit_pair_df.sort_values(by='Buy_Date', inplace= True) # 동일한 날짜 별로 정렬, 오름차순.
    profit_pair_df.reset_index(inplace=True)        
    profit_pair_df.to_csv("profitlog_pair_total_2021.csv",encoding='UTF-8-sig') 


    buy_date_no_duplicate = list(dict.fromkeys(profit_pair_df['Buy_Date'])) # 개장일 중 중복 없는 매수 일 리스트 생성. 
    pred_dates_dict=dict(zip(pred_Dates, [0 for x in range(len(pred_Dates))])) # 일 별 수익률 저장하는 딕셔너리
    
    total_agg=0
    top20sum=0
    day_list=dict()
    max_profit=0
    min_profit=100

    for idx in range(len(buy_date_no_duplicate)): # date list의 날짜들 (=매수일들)
        temp = profit_pair_df.loc[profit_pair_df['Buy_Date']==buy_date_no_duplicate[idx]] # 매수일이 같은 것들만 get

        top20 = temp.nlargest(20, ['Buy_Prob']) # buy probability 상위 20개 정렬'
        
        max_profit=(top20['Pair_수수료_수익률'].max()) # 상위 20개 종목 중 가장 높은 수익률
       
        
        min_profit_list= [i for i in top20['Pair_수수료_수익률'] if i > 0] # 상위 20개 종목 중 매수>매도이면서 가장 낮은 수익률
        for y in min_profit_list:
            if y<min_profit:
                min_profit=y
 
        count=0
        
        for top20index in top20.index:
            rising_rate = (top20['Sell_Price'][top20index]-top20['Buy_Price'][top20index])/top20['Buy_Price'][top20index] *100

            if rising_rate >= 10: # 매도 종가가 매수 종가보다 10%이상 상승했다면
                top20sum += top20['Pair_수수료_수익률'][top20index]
                count+=1
                day_count=list(pred_dates_dict).index( top20['Sell_Date'][top20index] )-list(pred_dates_dict).index( top20['Buy_Date'][top20index]) #매도> 매수 buy-sell 기간 차이. 
                day_list[buy_date_no_duplicate[idx]+str(count)]= [ top20['Buy_Date'][top20index],top20['Sell_Date'][top20index], day_count,  count,len(top20), max_profit, min_profit, ]
            else:
                pass
        #print(len(top20)) 문제점 1: count 길이로 나누면 조건을 만족하는 count 개수가 적어서  infinite number 발생. 
        
        top20sum = round(top20sum / len(top20),3)
        

        pred_dates_dict[buy_date_no_duplicate[idx]]=top20sum
        

    
    day_list_df = pd.DataFrame(day_list.values(),columns=['Buy_Date','Sell_Date','day_diff','T0P20-매도>매수-거래횟수', 'TOP20거래횟수' ,'max_profit','min_profit', ])
    print(day_list_df)
    day_list_df.to_csv("2021_analysis.csv", encoding='UTF-8 sig')        
    #keyList=sorted(pred_dates_dict.keys())

    

    """
    for index, key in enumerate(pred_dates_dict): # key의 인덱스, key 값
        if pred_dates_dict[key] ==0:
            pred_dates_dict[key] = pred_dates_dict[keyList[index-1]]
    """

    

    
    aggre_dict = dict(zip(pred_Dates, [0 for x in range(len(pred_Dates))])) # 누적 수익률을 저장하는 딕셔너리.
 

    for key,value in pred_dates_dict.items():
        total_agg+=value
        aggre_dict[key]=total_agg
   

    values=list(aggre_dict.values()) # 누적값 딕셔너리에서 value load
    total= values # 누적
    graph_result={'Buy_Date': aggre_dict.keys(), '누적_수익률':total}
    graph_result_csv = pd.DataFrame( graph_result)
    graph_result_csv.to_csv('graph_profit_total_2021.csv', encoding='UTF-8 sig')

    # 수익률 그래프 그리기. 
    plt.figure(figsize = (20,10))
    plt.xlabel('Date')
    plt.ylabel('Profit')
    plt.title('Yolo_Prediction_Profit_Rate')
    ax = plt.gca()
    formatter = mdates.DateFormatter("%Y-%m-%d")
    ax.xaxis.set_major_formatter(formatter)
    locator = mdates.MonthLocator()
    
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_minor_locator(mdates.DayLocator())

    #print(aggre_dict.values())
    

    
    x_values = [dt.strptime(d, "%Y-%m-%d").date() for d in aggre_dict.keys()] # x-value
    plt.plot(x_values, total, label = 'total',color='red')

    plt.plot(x_values, pred_dates_dict.values(),label='Day', color='blue')
    plt.legend(loc='upper left')
    plt.yticks(np.arange(0, 1000,15))

    plt.grid(True)
#    plt.gcf().autofmt_xdate()
    plt.savefig('graph_total_2021.png')
    #print("plus_count: "+str(plus_count))
    #print("minus_count: "+str(minus_count))
    
yoloboom()

