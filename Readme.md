# 系统使用说明

## 文件夹设置

在使用系统前，请您确保您的文件夹设置如下：
1. 创建文件夹"dir"，下设子文件夹"data"和"your_repo".
2. 将数据文件`stk_daily.feather`放在“data”文件夹下。
3. 请`test.ipynb`和`mysystem.py`放在"your_repo"文件夹下。
4. 在"your_repo"文件夹下创建"newdata"、"Results"和"Stocks_Chosen"三个子文件夹。
5. 将本repo中"newdata"中的数据下载并放进您的同名文件夹中。
6. 每次运行回测前，请确保"Stocks_Chosen"文件夹为空文件夹，否则"Stocks_Chosen"文件夹中将残留前次回测的输出结果。

总之，您的文件夹设置应如下所示：
| dir/ |            |                   |                    |
|------|------------|-------------------|--------------------|
|      | data/      |                   |                    |
|      |            | stk_daily.feather |                    |
|      | your_repo/ |                   |                    |
|      |            | newdata/          |                    |
|      |            |                   | hs300.csv          |
|      |            |                   | risk_free_rate.csv |
|      |            | Results/          |                    |
|      |            | Stocks_Chosen/    |                    |
|      |            | test.ipynb        |                    |
|      |            | mysystem.py       |                    |
|      |            | Readme.md         |                    |

## python库说明

在使用系统前，请确保您的python环境中安装了：  
    pandas  
    numpy  
    datetime  
    matplotlib  
    os  

## 策略输入说明

请您运行`test.ipynb`并按照“xx xxxx-xx-xx xxxx-xx-xx xx"的格式输入投资策略、回测的时间区间
以及调仓频率（单位为天）。  
若不输入调仓频率，则调仓频率默认为1天，即每天调仓一次。  
输入示例：3日动量策略 2020-01-02 2022-12-30 20  
输入完成后请按下“Enter”键。  
回测时间区间为“2020-01-02”至“2022-12-30”。  
策略种类限制为n日动量策略（其他策略待开发中……）  

## 回测时间补充说明

1. 建议回测时间为“2020-01-02”至“2022-12-30”，如若超出，运行时间相对较长。
2. 建议调仓频率大于15天，策略表现较好。
3. 若回测时间超出以上范围，请您仔细阅读`test.ipynb`中的说明，并按照指引：
    1. 保证您的电脑已经安装了WindPy，并且已经成功登录WindPy；
    2. 在同文件夹下的“mysystem.py”文件中，将19-20行的
       from WindPy import *
       w.start()
       两行代码取消注释（即将#删去）；
    3. 将`test.ipynb`该段注释中的以下两行代码取消注释（即将#删去）
       ms.get_hs300()
       ms.get_bond()  
    完成以上步骤后，请您继续阅读下一段指引并运行回测。

## 结果输出

查收回测结果，您可以：
1. 直接在`test.ipynb`中运行代码，并查收结果
2. 在"Stocks_Chosen"文件夹下，查收每日的持仓情况。
3. 在“outputs”文件夹下，查收净值曲线和收益率曲线。
4. 在“outputs”文件夹下，查收年化收益、年化波动、超额收益、夏普率、信息比率、胜率、最大回撤、alpha和beta等策略指标。

## 数据说明

1. 股票日行情 `stk_daily.feather`

|  列名  |      含义    |
|--------|-------------|
| stk_id | 股票ID       |
| date   | 日期         |
| open   | 开盘价       |
| high   | 最高价       |
| low    | 最低价       |
| close  | 收盘价       |
| volume | 成交量       |
| amount | 成交额       |
| cumadj | 累积复权因子 |

2. 沪深300日收盘价/收益率 `hs300.csv`

|  列名  |      含义    |
|--------|-------------|
| DATE   | 日期        |
| CLOSE  | 每日收盘价   |
| RETURNS| 对数收益率   |

3. 3个月国债收益率 `risk_free_rate.csv`

|  列名  |      含义    |
|--------|-------------|
| DATE   | 日期        |
| CLOSE  | 每日收盘价   |
| RETURNS| 对数收益率   |

4. Stocks_Chosen中的调仓文件`日期.csv`

|      列名      |       含义      |
|----------------|----------------|
| 调整日期        | 调仓日期        |
| 证券代码        | 证券代码        |
| 持仓份数        | 持仓份数        |
| 当日收盘价      | 当日收盘价      |
| n日内平均收益率 |  n日内平均收益率 |
| n日内平均交易量 |  n日内平均交易量 |
