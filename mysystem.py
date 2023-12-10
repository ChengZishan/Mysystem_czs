# ==================================
#        程子珊 2000015458
#  当代量化交易系统的原理与实现大作业
# ==================================

# ==========================================================================================================
# This python file is the main sysytem I designed, in which there are all the functions used in the jupyter.
# This file will be imported at the beginning in the jupyter.
# Basic introduction:
#   该系统的设计目的是：针对用户选择的不同投资策略，输入的回测的时间区间，对策略的效果进行回测。
#   该系统的主要功能包括：
#       1. 读取数据
#       2. 数据处理
#       3. 动量策略实现
#       4. 回测
#       5. 指标和csv文件生成
#       6. 画图和结果输出
#       7. 单独查看策略指标
#       8. 其他时间区间函数补充
#   该系统的使用方法：
#       1. 在jupyter中导入该文件
#       2. 调用get_data()函数，输入策略种类和回测的时间区间，得到一个dataframe，包含了回测时间区间内的全市场OHLCV数据
#       3. 调用rate_vol_cal()函数，对dataframe进行处理，计算日收益率、n日均值收益率和n日均值交易量
#       4. 调用get_tables()函数，得到三个dataframe，分别是：
#           1. 时间为index，股票id为columns，close为values的df
#           2. 时间为index，股票id为columns，n日均值交易量为values的df
#           3. 时间为index，股票id为columns，n日均值收益率为values的df
#       5. 调用do_test()函数，对策略进行回测，得到一个列表，包含了每个调仓日的总资产
#       6. 调用get_csv()函数，将文件夹Stocks_Chosen中的所有csv文件合并为一个csv文件
#       7. 调用net_value()函数，通过money_list计算策略的净值曲线
#       8. 调用returns_cur()函数，通过money_list计算策略的收益率曲线
#       9. 调用benchmark_cur()函数，从文件夹newdata中读取沪深300的数据
#       10. 调用benchmark_annual_return()函数，计算基准的年化收益率(沪深300)
#       11. 调用benchmark_net_value()函数，计算基准的净值曲线
#       12. 调用annual_returns()函数，通过money_list计算策略的年化收益率
#       13. 调用annual_volatility()函数，通过returns_cur计算策略的年化波动率
#       14. 调用max_drawdown()函数，通过money_list计算策略的最大回撤
#       15. 调用risk_free_rate()函数，用3个月国债的年化收益率作为无风险利率
#       16. 调用sharpe_ratio()函数，通过annual_returns和annual_volatility计算策略的夏普比率
#       17. 调用information_ratio()函数，通过returns_cur计算策略的信息比率
#       18. 调用win_rate()函数，通过returns_cur计算策略的胜率
#       19. 调用alpha()函数，通过annual_returns和annual_volatility计算策略的alpha
#       20. 调用beta()函数，通过returns_cur计算策略的beta
#       21. 调用excess_return()函数，计算策略的超额收益率
#       22. 调用plot_net_value()函数，画出策略和基准的净值曲线
#       23. 调用plot_return_cur()函数，画出策略的收益率曲线和沪深300的收益率曲线
#       24. 调用output_csv()函数，将计算出的各项指标输出为csv文件
# ==========================================================================================================

import pandas as pd
import numpy as np
import datetime
import os
import glob
import matplotlib.pyplot as plt
# from WindPy import *
# w.start()

# 1. 读取数据

# 取出字符串strategy中的数字并转换为int
def get_strategy_day(strategy):
    return int(''.join(filter(str.isdigit, strategy)))

# 计算start_date和end_date之间的交易日天数
def get_delta_date(start_date, end_date):
    delta_date = datetime.datetime.strptime(end_date,'%Y-%m-%d')-datetime.datetime.strptime(start_date,'%Y-%m-%d')
    delta_date = delta_date.days
    return delta_date

# 时间区间的起始日期和结束日期
start_date = ''
end_date = ''

# 反转策略的天数
strategy_day = 0

# 调仓频率
change_day = 1

# 策略种类
strategy = ''

# 数据集
data = pd.DataFrame()

# 读取指定时间区间内数据的函数
def get_data():
    # Args:
    #   None
    # Return:
    #   strategy：一个字符串，策略名字
    #   data：一个dataframe，在指定时间区间内的全市场OHLCV数据，格式与stk_daily.feather一致

    global start_date
    global end_date
    global strategy_day
    global change_day
    global strategy
    global data

    print('请按照所给的格式输入策略种类和回测的时间区间：')
    date_input = input()
    info_list = date_input.split()
    strategy = info_list[0]
    strategy_day = get_strategy_day(strategy)
    start_date = info_list[1]
    end_date = info_list[2]
    if (len(info_list) == 4):
        change_day = int(info_list[3])

    # 计算输入的时间区间的长度
    delta_date = get_delta_date(start_date, end_date)

    # 判断输入的时间区间是否符合格式要求
    while (len(start_date) != 10 or len(end_date) != 10 or delta_date <= strategy_day):
        if len(start_date) != 10 or len(end_date) != 10:
            print('时间区间输入格式错误，请重新输入：')
        elif delta_date <= strategy_day:
            print('时间区间过短，请重新输入：')
        print('请按照所给的格式输入策略种类和回测的时间区间：')
        date_input = input()
        info_list = date_input.split()
        strategy = info_list[0]
        strategy_day = get_strategy_day(strategy)
        start_date = info_list[1]
        end_date = info_list[2]
        if (len(info_list) == 4):
            change_day = int(info_list[3])
        delta_date = get_delta_date(start_date, end_date)
    
    print("数据读取成功，正在进行回测，请稍候...")

    # 抽出数据集中指定时间区间的数据
    if start_date >= '2020-01-02' and end_date <= '2022-12-30':
        data = pd.read_feather('../data/stk_daily.feather')

        # 通过date对数据集重新排序，取出时间区间内的所有数据
        data = data.sort_values('date')
        data.index = [i for i in range(data.shape[0])]
        # print(data)
        s = data['date']
        start_index = s.where(s==start_date).first_valid_index()
        end_index = s.where(s==end_date).last_valid_index()
        # print(start_index,' ',end_index)
        data = data[start_index:end_index]

        # 用股票id和时间对已取出的数据重新排序，变为原数据集的格式
        data = data.sort_values(['stk_id','date'])
        data.index = [i for i in range(data.shape[0])]
        data = data.drop(['amount','cumadj'],axis=1)
    else:
        # 使用wind API取出A股市场的全部股票id
        A_stocks=w.wset("sectorconstituent","date="+end_date+";sectorid=a001010100000000").Data
        assert A_stocks, "wset函数取出的数据集为空"  # API异常处理
        A_stocks = A_stocks[1]

        factor= ["OPEN","HIGH","LOW","CLOSE","VOLUME"]  #需要的参数

        # 对所有股票在时间区间内提取数据，并拼接为df data
        data = pd.DataFrame()
        for i in range(1,len(A_stocks)):
            error,data_tmp=w.wsd(A_stocks[i],factor,start_date,end_date,'priceAdj=F',usedf = True)
            assert error == 0, "API数据提取错误，ErrorCode={}".format(error)  # API异常处理
            id_name = [A_stocks[i]]*data_tmp.shape[0]
            data_tmp.insert(1,'stk_id',id_name)
            data= pd.concat([data, data_tmp])

    return strategy,data


# 2. 数据处理

# 日收益率计算，形成新的一列‘In_rate_returns’代表日收益率
# 平均收益率计算，形成新的一列‘n-days_rate_returns’代表n日均值收益率
# 平均交易量计算，形成新的一列‘n-days_volume’代表n日均值交易量
# 返回一个字典，不同的股票分别计算日收益率，n日均收益率和n日均交易量后的df
# 返回一个新的df，在原数据集的基础上增加了日收益率、n日均值收益率和n日均值交易量三列

data_dic = dict()
def rate_vol_cal():
    # Args:
    #   None
    # Returns:
    #   data_dic: 一个字典，不同的股票分别计算完日收益率后的df
    #   new_data: 一个df，所有股票计算完日收益率后的df
    
    global data_dic
    global data
    # 用股票id对数据集进行分组
    data_grouped = data.groupby('stk_id')
    data_list = []

    # 对每只股票计算日收益率、n日均值收益率和n日均值交易量
    for stk_id, stk_data in data_grouped:
        stk_data['ln_rate_returns'] = pd.Series(np.log(stk_data.close/stk_data.close.shift(1)),index=stk_data.index)
        stk_data = stk_data.bfill()
        stk_data[str(strategy_day)+'-days_rate_returns'] = stk_data['ln_rate_returns'].rolling(strategy_day).mean()
        stk_data[str(strategy_day)+'-days_volume'] = stk_data['volume'].rolling(strategy_day).mean()
        data_dic.update({stk_id:stk_data})
        data_list.append(stk_data)
    new_data = pd.concat(data_list)
    data = new_data

    return data_dic,new_data

# 用时间做index，股票id做columns，open、close、n日均值交易量、日收益率、n日均值收益率分别取出df
close_data = pd.DataFrame()
mean_volume_data = pd.DataFrame()
mean_return_data = pd.DataFrame()

def get_tables():
    # Args:
    #   None
    # Returns:
    #   close_data: 一个df，时间为index，股票id为columns，close为values
    #   mean_volume_data: 一个df，时间为index，股票id为columns，n日均值交易量为values
    #   mean_return_data: 一个df，时间为index，股票id为columns，n日均值收益率为values

    global close_data
    global mean_volume_data
    global mean_return_data
    global data
    
    close_data = pd.pivot_table(data,index='date',columns='stk_id',values='close')
    mean_volume_data = pd.pivot_table(data,index='date',columns='stk_id',values=str(strategy_day)+'-days_volume')
    mean_return_data = pd.pivot_table(data,index='date',columns='stk_id',values=str(strategy_day)+'-days_rate_returns')

    return close_data,mean_volume_data,mean_return_data


# 3. 动量策略实现

# 存储每个调仓日的总资产
money_list = []

# 实现一个n日动量策略，返回一个列表，包含了今天所选的购入的股票代码
# 在文件夹中Stocks_Chosen中生成一个名为“当日日期.csv”的文件，记录了每天的持仓情况
def choose_stk(day, last = None, First_day = False):
    # Args:
    #   day: 一个datatime格式的日期，表示今天的日期
    #   last: 一个datatime格式的日期，表示上一天的日期
    #   First_day: 一个布尔值，表示是否是第一天，如果是第一天，则不需要读取上一天的持仓情况
    # Returns:
    #   stk_list: 一个列表，包含了今天所选的购入的股票代码
    #   start_money: 一个float，表示今天的总资产
    
    global money_list
    # 读取上一天的持仓情况, 并计算经过一天后资产的变化
    if not First_day:
        df = pd.read_csv('Stocks_Chosen/' + str(last.date()) + '.csv')
        df['当日收盘价'] = close_data.loc[day][df['证券代码']].values
        start_money = sum(df['持仓份数'] * df['当日收盘价'])
    else:
        start_money = 10000
    
    # 如果总资产小于或等于0，则将总资产置为0
    if start_money <= 0:
        start_money = 0
    
    money_list.append(start_money)

    # 选出今天的股票列表
    return_sort = mean_return_data.loc[day].sort_values(ascending=False).dropna()
    return_chosen = mean_volume_data[list(return_sort.index[0:100])].loc[day].sort_values(ascending=False).dropna()
    stk_list = list(return_chosen.index[0:50])

    # 生成今天的持仓情况csv文件
    columns = ['调整日期', '证券代码', '持仓份数', '当日收盘价', str(strategy_day)+'日内平均收益率', str(strategy_day)+'日内平均交易量']
    df_today = pd.DataFrame(columns=columns)
    df_today['调整日期'] = [day]*50
    df_today['证券代码'] = stk_list
    df_today['当日收盘价'] = close_data.loc[day][stk_list].values
    df_today['持仓份数'] = df_today['当日收盘价'].apply(lambda x: int(start_money/50/x))
    df_today[str(strategy_day)+'日内平均收益率'] = mean_return_data.loc[day][stk_list].values
    df_today[str(strategy_day)+'日内平均交易量'] = mean_volume_data.loc[day][stk_list].values
    df_today.to_csv('Stocks_Chosen/' + str(day.date()) + '.csv', index=False)

    return stk_list, start_money

# 如果当日不调仓，则持续持有上一日的股票，计算资产的变化
def hold_stk(day, last):
    # Args:
    #   day: 一个datatime格式的日期，表示今天的日期
    #   last: 一个datatime格式的日期，表示上一天的日期
    # Returns:
    #   stk_list: 一个列表，包含了今天所选的购入的股票代码
    #   start_money: 一个float，表示今天的总资产
    
    global money_list
    # 读取上一天的持仓情况, 并计算经过一天后资产的变化
    df = pd.read_csv('Stocks_Chosen/' + str(last.date()) + '.csv')
    df['当日收盘价'] = close_data.loc[day][df['证券代码']].values
    start_money = sum(df['持仓份数'] * df['当日收盘价'])
    stk_list = list(df['证券代码'])
    money_list.append(start_money)

    return stk_list, start_money


# 4. 回测

# 计算回测区间内的所有交易日, 以列表形式存储
date_list = []

def get_date_list():
    # Args:
    #   None
    # Returns:
    #   date_list: 一个列表，包含了回测时间区间内的所有交易日
    
    global start_date
    global end_date
    global date_list
    global mean_return_data

    date_list = mean_return_data.index
    # 找出date_list中第一个大于等于start_date的日期
    for i in range(len(date_list)):
        if date_list[i].date() >= datetime.datetime.strptime(start_date,'%Y-%m-%d').date():
            start_datetime = date_list[i].date()
            start_dateindex = i
            break
    # 找出date_list中最后一个小于等于end_date的日期
    for i in range(len(date_list)-1,-1,-1):
        if date_list[i].date() <= datetime.datetime.strptime(end_date,'%Y-%m-%d').date():
            end_datetime = date_list[i].date()
            end_dateindex = i
            break

    date_list = date_list[start_dateindex:end_dateindex+1]

    return date_list

# 对时间区间内的每一天执行策略
# 如果是调仓日，则调用choose_stk函数
# 如果不是调仓日，则调用hold_stk函数
def do_test():
    # Args:
    #   None
    # Returns:
    #   None

    global date_list
    global change_day
    global end_date

    last = date_list[0]
    for i in range(len(date_list)):
        if i == 0:
            stk_list, start_money = choose_stk(date_list[i], last = None, First_day = True)
        elif i % change_day == 0:
            stk_list, start_money = choose_stk(date_list[i], last = last, First_day = False)
            last = date_list[i]
        else:
            stk_list, start_money = hold_stk(date_list[i], last)

        # 如果总资产小于或等于0，则回测在当天提前结束
        if start_money == 0:
            print('总资产降为0，','回测在', date_list[i].date(), '提前结束')
            print('以下为本次回测结果：')
            date_list = date_list[0:i+1]
            end_date = str(date_list[-1].date())
            break

    return


# 5. 指标和csv文件生成

# 将文件夹Stocks_Chosen中的所有csv文件合并为一个csv文件
# 生成的csv文件名为“开始日期_to_结束日期.csv”
def get_csv():
    # Args:
    #   None
    # Returns:
    #   None
    
    path = 'Stocks_Chosen'
    all_filenames = [os.path.join(path,i) for i in os.listdir(path)]
    combined_csv = pd.concat([pd.read_csv(f) for f in all_filenames ])
    combined_csv.to_csv( os.path.join(path, "{}_to_{}.csv".format(start_date, end_date)), index=False)

    return

# 通过money_list计算策略的净值曲线
net_value = []
def s_net_value():
    # Args:
    #   None
    # Returns:
    #   net_value: 一个列表，包含了每个调仓日的净值
    
    global net_value
    global money_list

    for i in range(len(money_list)):
        net_value.append(money_list[i]/10000)

    return net_value

# 通过money_list计算策略的收益率曲线
returns_cur = []
def returns_cur():
    # Args:
    #   None
    # Returns:
    #   returns: 一个列表，包含了每个调仓日的收益率

    global money_list
    global returns_cur
    
    returns_cur = [0 for i in range(len(money_list)-1)]
    # print(len(returns_cur))
    for i in range(len(money_list)-1):
        if money_list[i] == 0:
            break
        returns_cur[i] = ((money_list[i+1]-money_list[i])/money_list[i])

    return returns_cur

start_index = 0
end_index = 0
hs300 = pd.DataFrame()

# 从文件夹newdata中读取沪深300的数据
def benchmark_cur():
    # Args:
    #   None
    # Returns:
    #   benchmark_cur: 一个列表，包含了每个调仓日的基准收益率
    
    global start_index
    global end_index
    global start_date
    global end_date
    global date_list
    global hs300

    benchmark = pd.read_csv('newdata/hs300.csv')
    benchmark = benchmark.iloc[:-1,:]
    benchmark.set_index(benchmark.iloc[:,0], inplace=True)
    benchmark.index = [datetime.datetime.strptime(str(i),'%Y-%m-%d') for i in benchmark.index]
    benchmark.index.name = 'DATE'
    del benchmark['DATE']
    # print(benchmark)
    # 找出给定时间区间中第一个大于等于date_list[0]的日期
    for i in range(len(benchmark)):
        if benchmark.index[i] >= date_list[0]:
            start_index = i
            break
    # 找出给定时间区间中最后一个小于等于date_list[-1]的日期
    for i in range(len(benchmark)-1,-1,-1):
        if benchmark.index[i] <= date_list[-1]:
            end_index = i
            break
    benchmark = benchmark[start_index:end_index+1]

    hs300 = benchmark

    return benchmark

# 计算基准的年化收益率(沪深300)
benchmark_annual_return = 0
def benchmark_annual_return():
    # Args:
    #   None
    # Returns:
    #   annual_return: 一个浮点数，表示基准的年化收益率
    
    global hs300
    global benchmark_annual_return
    hs300 = benchmark_cur()

    benchmark_annual_return = (hs300['CLOSE'].iloc[-1]/hs300['CLOSE'].iloc[0])**(252/len(hs300))-1

    return benchmark_annual_return

# 计算基准的净值曲线
benchmark_net_value = []
def s_benchmark_net_value():
    # Args:
    #   None
    # Returns:
    #   benchmark_net_value: 一个列表，包含了每个调仓日的基准净值
    
    global hs300
    global benchmark_net_value

    for i in range(len(hs300)):
        benchmark_net_value.append(hs300['CLOSE'].iloc[i]/hs300['CLOSE'].iloc[0])

    return benchmark_net_value

# 通过money_list计算策略的年化收益率
annual_returns = 0
def annual_returns():
    # Args:
    #   None
    # Returns:
    #   annual_returns: 一个浮点数，代表了策略的年化收益率
    
    global money_list
    global annual_returns

    annual_returns = (money_list[-1]/money_list[0])**(252/len(money_list))-1

    return annual_returns

# 通过returns_cur计算策略的年化波动率
annual_volatility = 0
def annual_volatility():
    # Args:
    #   None
    # Returns:
    #   annual_volatility: 一个浮点数，代表了策略的年化波动率
    
    global returns_cur
    global annual_volatility

    annual_volatility = np.std(returns_cur)*np.sqrt(252)

    return annual_volatility

# 通过money_list计算策略的最大回撤
max_drawdown = 0
def s_max_drawdown():
    # Args:
    #   None
    # Returns:
    #   max_drawdown: 一个浮点数，代表了策略的最大回撤
    
    global money_list
    global max_drawdown

    for i in range(len(money_list)):
        for j in range(i+1,len(money_list)):
            if money_list[j] == 0:
                break
            drawdown = (money_list[j]-money_list[i])/money_list[i]
            if drawdown < max_drawdown:
                max_drawdown = drawdown
        if money_list[i] == 0:
            break

    return max_drawdown

# 用3个月国债的年化收益率作为无风险利率
risk_free_rate = 0
def risk_free_rate():
    # Args:
    #   None
    # Returns:
    #   risk_free_rate: 一个浮点数，代表了无风险利率
    
    global start_index
    global end_index
    global risk_free_rate

    bond = pd.read_csv('newdata/risk_free_rate.csv')
    bond = bond.iloc[start_index:end_index+1,:]
    risk_free_rate = (bond['CLOSE'].iloc[-1]/bond['CLOSE'].iloc[0])**(252/len(bond))-1

    return risk_free_rate

# 通过annual_returns和annual_volatility计算策略的夏普比率
sharpe_ratio = 0
def sharpe_ratio():
    # Args:
    #   None
    # Returns:
    #   sharpe_ratio: 一个浮点数，代表了策略的夏普比率
    
    global sharpe_ratio
    global risk_free_rate
    global annual_returns
    global annual_volatility

    sharpe_ratio = (annual_returns-risk_free_rate)/annual_volatility

    return sharpe_ratio

# 通过returns_cur计算策略的信息比率
information_ratio = 0
def information_ratio():
    # Args:
    #   None
    # Returns:
    #   information_ratio: 一个浮点数，代表了策略的信息比率
    
    global information_ratio
    global risk_free_rate
    global annual_returns
    global returns_cur

    information_ratio = (annual_returns-risk_free_rate)/np.std(returns_cur)

    return information_ratio

# 通过returns_cur计算策略的胜率
win_rate = 0
def win_rate():
    # Args:
    #   None
    # Returns:
    #   win_rate: 一个浮点数，代表了策略的胜率
    
    global win_rate
    global returns_cur

    win_rate = len([i for i in returns_cur if i > 0])/len(returns_cur)

    return win_rate

# 通过annual_returns和annual_volatility计算策略的alpha
alpha = 0
def alpha():
    # Args:
    #   None
    # Returns:
    #   alpha: 一个浮点数，代表了策略的alpha

    global alpha
    global risk_free_rate
    global annual_returns
    global annual_volatility
    
    alpha = (annual_returns-risk_free_rate)-risk_free_rate*annual_volatility

    return alpha

# 通过returns_cur计算策略的beta
beta = 0
def beta():
    # Args:
    #   None
    # Returns:
    #   beta: 一个浮点数，代表了策略的beta
    
    global beta
    global returns_cur
    global hs300

    beta = np.cov(returns_cur, hs300['RETURNS'][1:])/np.var(hs300['RETURNS'][1:])

    return beta

# 计算策略的超额收益率
excess_return = 0
def excess_return():
    # Args:
    #   None
    # Returns:
    #   excess_return: 一个浮点数，代表了策略的超额收益率
    
    global excess_return
    global annual_returns
    global benchmark_annual_return

    excess_return = annual_returns-benchmark_annual_return

    return excess_return


# 6. 画图和结果输出

# 设置所有的画图参数字典，方便修改和取用
def plot_parameter():
    font_dic = {'sans-serif':'KaiTi','fontsize':10,'fontweight':'bold','fontstyle':'italic',\
                'dpi':500,'linestyle':'-','linewidth':1,'marker':'.','markersize':1,'color':'black',\
                    'alpha':1,'figsize':(10,5),'facecolor':'white','edgecolor':'white','frameon':True}
    def set_font(font_dic):
        plt.rcParams['font.sans-serif'] = font_dic['sans-serif']
        plt.rcParams['font.size'] = font_dic['fontsize']
        plt.rcParams['font.weight'] = font_dic['fontweight']
        plt.rcParams['font.style'] = font_dic['fontstyle']
        plt.rcParams['figure.dpi'] = font_dic['dpi']
        plt.rcParams['lines.linestyle'] = font_dic['linestyle']
        plt.rcParams['lines.linewidth'] = font_dic['linewidth']
        plt.rcParams['lines.marker'] = font_dic['marker']
        plt.rcParams['lines.markersize'] = font_dic['markersize']
        plt.rcParams['lines.color'] = font_dic['color']
        plt.rcParams['figure.figsize'] = font_dic['figsize']
        plt.rcParams['figure.facecolor'] = font_dic['facecolor']
        plt.rcParams['figure.edgecolor'] = font_dic['edgecolor']
        plt.rcParams['figure.frameon'] = font_dic['frameon']
        plt.rcParams['axes.unicode_minus'] = False
    set_font(font_dic)
    return font_dic

# 画出策略和基准的净值曲线
def plot_net_value():
    # Args:
    #   None
    # Returns:
    #   None
    
    global net_value
    global benchmark_net_value
    global date_list
    global hs300

    # 设置画图参数
    font_dic = plot_parameter()

    plt.plot(date_list, net_value)
    plt.plot(hs300.index, benchmark_net_value)
    plt.legend(['策略净值','基准净值'], loc='upper right')
    plt.title('净值曲线')
    plt.xlabel('日期')
    plt.ylabel('策略/基准净值')
    
    # 将净值曲线存储为文件夹Results下的png文件
    plt.savefig('Results/净值曲线.png')
    plt.show()

    return

# 画出策略的收益率曲线和沪深300的收益率曲线
def plot_return_cur():
    # Args:
    #   None
    # Returns:
    #   None
    
    global returns_cur
    global date_list
    global hs300

    # 设置画图参数
    font_dic = plot_parameter()

    hs300_returns = list(hs300['RETURNS'][1:])
    plt.plot(date_list[1:], returns_cur)
    plt.plot(date_list[1:], hs300_returns)
    plt.title('收益率曲线')
    plt.xlabel('日期')
    plt.ylabel('收益率')
    plt.legend(['策略收益率','沪深300收益率'], loc='upper right')
    
    # 将收益率曲线存储为文件夹Results下的png文件
    plt.savefig('Results/收益率曲线.png')
    plt.show()

    return

# 将计算出的各项指标输出为csv文件
def output_csv():
    # Args:
    #   None
    # Returns:
    #   None
    
    global annual_returns
    global annual_volatility
    global excess_return
    global max_drawdown
    global sharpe_ratio
    global information_ratio
    global win_rate
    global alpha
    global beta
    global benchmark_annual_return
    global risk_free_rate

    df = pd.DataFrame()
    df['策略年化收益率'] = [annual_returns]
    df['策略年化波动率'] = [annual_volatility]
    df['策略超额收益率'] = [excess_return]
    df['策略最大回撤'] = [max_drawdown]
    df['策略夏普比率'] = [sharpe_ratio]
    df['策略信息比率'] = [information_ratio]
    df['策略胜率'] = [win_rate]
    df['策略alpha'] = [alpha]
    df['策略beta'] = [beta]
    df['基准年化收益率'] = [benchmark_annual_return]
    df['无风险利率'] = [risk_free_rate]
    df.to_csv('Results/策略指标.csv', index=False)

    return

# 7. 单独查看策略指标

def get_net_value():
    # Args:
    #   None
    # Returns:
    #   df_net_value: 一个df，显示策略净值随时间的变化

    df_net_value = pd.DataFrame(index = date_list, columns = ['策略净值'])
    df_net_value.index.name = '日期'
    df_net_value['策略净值'] = net_value

    return df_net_value

def get_hs300_net_value():
    # Args:
    #   None
    # Returns:
    #   df_hs300_net_value: 一个df，显示沪深300净值随时间的变化

    df_hs300_net_value = pd.DataFrame(index = date_list, columns = ['沪深300净值'])
    df_hs300_net_value.index.name = '日期'
    df_hs300_net_value['沪深300净值'] = benchmark_net_value

    return df_hs300_net_value

def get_return_cur():
    # Args:
    #   None
    # Returns:
    #   df_return_cur: 一个df，显示策略收益率随时间的变化

    df_return_cur = pd.DataFrame(index = date_list[1:], columns = ['策略收益率'])
    df_return_cur.index.name = '日期'
    df_return_cur['策略收益率'] = returns_cur

    return df_return_cur

def get_hs300_return_cur():
    # Args:
    #   None
    # Returns:
    #   df_hs300_return_cur: 一个df，显示策略收益率随时间的变化

    df_hs300_return_cur = pd.DataFrame(index = date_list[1:], columns = ['沪深300收益率'])
    df_hs300_return_cur.index.name = '日期'
    df_hs300_return_cur['沪深300收益率'] = hs300['RETURNS'][1:]

    return df_hs300_return_cur

# 8. 其他时间区间函数补充

# 更新指定时间区间内的沪深300数据集
def get_hs300(start_date,end_date):
    # Args:
    #   start_date: 一个字符串，表示时间区间的开始日期
    #   end_date: 一个字符串，表示时间区间的结束日期
    # Returns:
    #   None

    error, hs300 = w.wsd("000300.SH", "close", start_date, end_date, "PriceAdj=F", usedf = True)
    assert error == 0, "API数据提取错误，ErrorCode={}".format(error)  # API异常处理
    hs300['RETURNS'] = pd.Series(np.log(hs300['CLOSE']/hs300['CLOSE'].shift(1)),index=hs300.index)
    hs300 = hs300.bfill()
    hs300.index = [datetime.datetime.strptime(str(i),'%Y-%m-%d') for i in hs300.index]
    hs300.index.name = 'DATE'
    hs300.to_csv('newdata/hs300.csv')

    return

# 更新指定时间区间内的3个月国债数据集
def get_bond(start_date,end_date):
    # Args:
    #   start_date: 一个字符串，表示时间区间的开始日期
    #   end_date: 一个字符串，表示时间区间的结束日期
    # Returns:
    #   None

    error, bond = w.wsd("H11006.CSI", "close", start_date, end_date, "PriceAdj=F", usedf = True)
    assert error == 0, "API数据提取错误，ErrorCode={}".format(error)  # API异常处理
    bond['RETURNS'] = pd.Series(np.log(bond['CLOSE']/bond['CLOSE'].shift(1)),index=bond.index)
    bond = bond.bfill()
    bond.index = [datetime.datetime.strptime(str(i),'%Y-%m-%d') for i in bond.index]
    bond.index.name = 'DATE'
    bond.to_csv('newdata/risk_free_rate.csv')

    return