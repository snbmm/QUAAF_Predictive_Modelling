import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from scipy.stats import norm
import datetime
from datetime import timezone
import yahooquery as yq
from scipy.optimize import curve_fit
import base64
import io
import warnings
warnings.filterwarnings("ignore")
import pytz

ny_timezone = pytz.timezone('America/New_York')
today = datetime.datetime.now(ny_timezone).date()

def concave_curve(x, a, b, c):
    return a * x**2 + b * x + c

def get_fridays(weeks = 10, format = "%Y-%m-%d"):
    fridays = []
    date = today + datetime.timedelta(days=(4-today.weekday()) % 7) # next Friday
    if date == today:
        date += datetime.timedelta(days = 7)
    one_year_later = today + datetime.timedelta(days=7*weeks)

    while date <= one_year_later:
        fridays.append(date.strftime(format))
        date += datetime.timedelta(days=7)
    return fridays

def get_option_week(option_symbols, check_put = False):
    dates = get_fridays(format='%y%m%d')
    for i in range(len(dates)):
        if dates[i] + check_put*'P' in option_symbols:
            return i
    return -1

date_format = "%Y-%m-%d"

def get_iv_plot(sym = 'AAPL', steps = 100, option_type = 'call', weeks = 4, rf = yq.Ticker('^TNX').history(period='1y')['adjclose'][-1]/100, trade_date = 3, curvefit_t = 2):

    # 获取标的资产历史价格数据
    borders = [{
              'selector': 'td, th, table', 
              'props'   : [ ('border', '1px solid lightgrey'), ('border-collapse', 'collapse'),('font-size', '0.5em')]
            }]
    tick = yf.Ticker(sym)
    hist_prices = tick.history(period='max')['Close']

    # 计算每个到期期限的隐含波动率
    prices = np.array(hist_prices)
    S = prices[-1]
    #date_list = ['2023-04-21', '2023-04-28', '2023-05-05', '2023-05-12', '2023-05-19', '2023-05-26', '2023-06-02', '2023-06-16', '2023-07-21', '2023-08-18', '2023-09-15', '2023-10-20', '2023-11-17', '2023-12-15', '2024-01-19', '2024-03-15', '2024-06-21', '2024-09-20', '2024-12-20', '2025-01-17', '2025-06-20', '2025-12-19']
    date_list = get_fridays(weeks)

    t = np.array(date_list)   # 到期期限（以年为单位
    #delta_t = [(lambda d: (datetime.datetime.strptime(d, date_format).date()- today).days/365)(d)\
    #            for d in date_list]

    K = np.array([S*i/steps for i in range(int(0.5*steps), int(1.5*steps)+1)])   # 行权价格
    iv = {}
    delta_t = []
    df_option_tables = {}
    utc_now = datetime.datetime.now(timezone.utc)
    for j in range(len(t)):
        try:
            if option_type == 'put':
                options = tick.option_chain(t[j]).puts
            else:
                options = tick.option_chain(t[j]).calls
            opt = options.loc[(utc_now - options['lastTradeDate']) / np.timedelta64(1, 'D') < trade_date]
        except Exception as e:
            print(e)
            continue
        delta_t.append((datetime.datetime.strptime(t[j], date_format).date()- today).days/365)
        iv[delta_t[-1]] = {'strike':[], 'sigma':[]}
        for idx in range(len(opt.strike.values)):
            opt_strike = opt.strike.values[idx]
            opt_price = opt.lastPrice.values[idx]
            sigma = 0.2   # 初始波动率猜测值
            for k in range(100):
                if option_type == 'put':
                    d1 = (np.log(S/opt_strike) + (rf - 0.5*sigma**2)*delta_t[-1]) / (sigma*np.sqrt(delta_t[-1]))
                    d2 = d1 - sigma*np.sqrt(delta_t[-1])
                    opt_price_bs = np.exp(-rf*delta_t[-1])*(opt_strike*norm.cdf(-d2) - S*norm.cdf(-d1))
                else:
                    d1 = (np.log(S/opt_strike) + (rf + 0.5*sigma**2)*delta_t[-1]) / (sigma*np.sqrt(delta_t[-1]))
                    d2 = d1 - sigma*np.sqrt(delta_t[-1])
                    opt_price_bs = S*norm.cdf(d1) - np.exp(-rf*delta_t[-1])*opt_strike*norm.cdf(d2)
                vega = S*np.sqrt(delta_t[-1])*norm.pdf(d1)
                sigma = sigma - (opt_price_bs - opt_price) / vega
            iv[delta_t[-1]]['strike'].append(opt_strike)
            iv[delta_t[-1]]['sigma'].append(sigma)
        opt['IV'] = iv[delta_t[-1]]['sigma']
        opt['IV'].max()
        opt['volume'] = opt['volume'].astype('int')
        df_option_tables[t[j]] = opt[['contractSymbol','strike','lastPrice','inTheMoney','volume','IV']].style.hide_index().set_table_styles(borders).bar(subset=['volume'], color='#5fba7d').bar(subset=['IV'], color='#d65f5f', vmin = opt['IV'].min(), vmax = opt['IV'].max()).render()

    figure, axis = plt.subplots(len(delta_t), 1, figsize=(8, 3*len(delta_t)), sharex= True, constrained_layout = True)
    figure.supxlabel('Strike price $')
    figure.supylabel('Implied Valotility')
    font = {'family' : 'normal',
        'size'   : 10}
    for i in range(len(delta_t)):
        #print(iv)
        df_iv_k = pd.DataFrame()
        df_iv_k['K'] = iv[delta_t[i]]['strike']
        df_iv_k['iv'] = iv[delta_t[i]]['sigma']
        df_iv_k.dropna(inplace=True)
        #print(df_iv_k)

        # Remove outlier using curvefit
        popt, pcov = curve_fit(concave_curve, df_iv_k['K'], df_iv_k['iv'])
        errors = np.abs(df_iv_k['iv'] - concave_curve(df_iv_k['K'], *popt))
        mad = np.median(errors)
        threshold = curvefit_t * mad
        df_iv_K_out = df_iv_k[errors >= threshold]
        df_iv_k_in = df_iv_k[errors < threshold]
        #print(df_iv_K_out)

        #Curve fit again without outliers
        popt, pcov = curve_fit(concave_curve, df_iv_k_in['K'], df_iv_k_in['iv'])

        x_fit = np.linspace(K[0], K[-1], 100)
        y_fit = concave_curve(x_fit, *popt)
        min_k = df_iv_k_in.loc[df_iv_k_in['iv'].idxmin()]['K']
        axis[i].set_title(f'Options experies within {round(delta_t[i]*365)} days', fontsize=13)
        axis[i].plot(df_iv_k_in['K'], df_iv_k_in['iv'], 'o', label=f'Inliers')
        axis[i].plot(df_iv_K_out['K'], df_iv_K_out['iv'], 'o', label=f'Outliers')
        axis[i].plot(x_fit, y_fit, '-.', label=f'Curve fit')
        axis[i].axvline(x=min_k, label = f'min_iv strike ${min_k:.2f}', linewidth = 1, linestyle = '-')
        axis[i].axvline(x=S, label = f'Current price ${S:.2f}', linewidth = 1, linestyle = '--')
        axis[i].legend(fontsize=10)
        #axis[1].plot(K, iv[i*len(K):(i+1)*len(K)], 'o-', label=f'{round(delta_t[i]*365)} days')
    plt.setp(axis, ylim=(0.0,2.0))
    img=io.BytesIO()
    plt.savefig(img,format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    return [plot_url, df_option_tables]


class Option_Optimizor():
    def __init__(self, ticker, prediction_iteration = 10000, option_choice = 0, period='5y'):
        self.ticker = ticker
        self.tick = yf.Ticker(ticker)
        self.prediction_iteration = prediction_iteration
        self.t = np.array(get_fridays())
        #print(self.t)
        self.option_choice = option_choice
        self.days  = np.busday_count(today, datetime.datetime.strptime(self.t[option_choice], date_format).date())
        self.hist_prices = self.tick.history(period=period)['Close']
        self.current_price = self.tick.info['currentPrice']
        self.end_price = [self.current_price]*self.prediction_iteration
        print("Before Stock {}'s {} days mean return is {}. std is {} using past {} data".format(ticker, self.days, self.hist_prices.pct_change(periods=self.days).mean(), self.hist_prices.pct_change(periods=self.days).std(),period))
        
        #self.end_price *= (np.random.normal(self.hist_prices.pct_change(periods=self.days).mean(),  self.hist_prices.pct_change(periods=self.days).std(),  size = self.prediction_iteration) +1)
        a = np.prod(np.random.normal(self.hist_prices.pct_change().mean(), self.hist_prices.pct_change().std(), size = [self.prediction_iteration, self.days]) +1, axis = 1)
        self.end_price *= a
        print("After Stock {}'s {} days mean return is {}. std is {} using past {} data".format(ticker, self.days, a.mean(), a.std(),period))

        #print(self.end_price)
        self.plot_first_10 = 10
        self.iteration_table = pd.DataFrame(columns=["Low Weight", "Sharpe Ratio 1", "High Weight", "Sharpe Ratio 2"])

    def option_sim(self, init_weight = 0.0, allocation_iteration = 1, num_options_in_portfolio=1, option_symbols = []):
        if option_symbols:
            df = self.tick.option_chain(self.t[self.option_choice]).puts
            #print(df['contractSymbol'])
            #print(option_symbols)
            opt = df[df['contractSymbol'].isin(option_symbols)]
            num_options_in_portfolio = len(option_symbols)
        else:
            num_options_in_portfolio = 1
            opt = self.tick.option_chain(self.t[self.option_choice]).puts.nlargest(num_options_in_portfolio, 'volume')

        total_earn = {}
        # Testing no option first
        put_weight = [init_weight] * num_options_in_portfolio
        for w in range(allocation_iteration):
            total_earn[str(put_weight)] = {'Return Rate': 0, 'Return Rate std': 0}
            total_return = []
            #print('prediction_iteration {}'.format(self.prediction_iteration))
            for i in range(self.prediction_iteration):
                cost = 0
                earn = 0
                #print(put_weight)
                for j in range(num_options_in_portfolio):
                    #print(list(opt['lastPrice'])[j])
                    #print(list(opt['strike'])[j] - end_price[i])
                    #print(max(list(opt['strike'])[j] - end_price[i], 0))
                    #if np.random.random() < 1/(self.prediction_iteration*10):
                    #    print("Option info: last price: {} strike: {}".format(list(opt['lastPrice'])[j], list(opt['strike'])[j]))
                    #    print("Stock curent price: {}, end price: {}, option cost: {}, earn: {}.".format(self.current_price, self.end_price[i], cost, earn))

                    cost += put_weight[j]*list(opt['lastPrice'])[j]
                    earn += put_weight[j]* max(list(opt['strike'])[j] - self.end_price[i], 0)
                #print(cost)
                #print(earn)
            #total_earn['Total Earn'].add(end_price[i] - current_price + earn - cost)
            #total_earn['Total Earn without option'].add(end_price[i] - current_price)
                total_return.append((self.end_price[i] - self.current_price + earn - cost)/self.current_price)


            total_earn[str(put_weight)]['Return Rate'] = np.mean(total_return)
            total_earn[str(put_weight)]['Return Rate std'] = np.std(total_return)
            total_earn[str(put_weight)]['Put Weight'] = put_weight[0]
            # set up next random allocation
            put_weight = np.random.random(num_options_in_portfolio)*10
        return {"total_earn":total_earn, "Sharpe Ratio":np.mean(total_return)/np.std(total_return)}


    def find_max_input_recursive(self, left, right, option_symbols = [], tolerance=1e-2):

        if right - left < tolerance:
            return (left + right) / 2 

        mid1 = (2 * left + right) / 3
        mid2 = (left + 2 * right) / 3

        f_mid1 = self.option_sim(init_weight = mid1, option_symbols = option_symbols)['Sharpe Ratio']
        f_mid2 = self.option_sim(init_weight = mid2, option_symbols = option_symbols)['Sharpe Ratio']
        self.iteration_table.loc[len(self.iteration_table.index)] = [mid1, f_mid1, mid2, f_mid2] 

        if f_mid1 < f_mid2:
            return self.find_max_input_recursive(mid1, right, option_symbols, tolerance)
        else:
            return self.find_max_input_recursive(left, mid2, option_symbols, tolerance)