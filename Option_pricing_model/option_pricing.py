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

def concave_curve(x, a, b, c):
    return a * x**2 + b * x + c

def get_fridays(weeks = 4):
    fridays = []
    date = datetime.date.today() + datetime.timedelta(days=(4-datetime.date.today().weekday()) % 7) # next Friday
    if date == datetime.date.today():
        date += datetime.timedelta(days = 7)
    one_year_later = datetime.date.today() + datetime.timedelta(days=7*weeks)

    while date <= one_year_later:
        fridays.append(date.strftime("%Y-%m-%d"))
        date += datetime.timedelta(days=7)
    return fridays

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
    #delta_t = [(lambda d: (datetime.datetime.strptime(d, date_format).date()- datetime.date.today()).days/365)(d)\
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
        df_option_tables[t[j]] = opt[['contractSymbol','strike','lastPrice','inTheMoney','volume']].style.hide_index().set_table_styles(borders).bar(subset=['volume'], color='#d65f5f').render()
        delta_t.append((datetime.datetime.strptime(t[j], date_format).date()- datetime.date.today()).days/365)
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

    # 绘制波动率微笑曲
    figure, axis = plt.subplots(len(delta_t), 1, figsize=(8, 3*len(delta_t)), sharex= True, constrained_layout = True)
    figure.supxlabel('Strike price $')
    figure.supylabel('Implied Valotility')
    font = {'family' : 'normal',
        'size'   : 10}
    for i in range(len(delta_t)):
        print(iv)
        df_iv_k = pd.DataFrame()
        df_iv_k['K'] = iv[delta_t[i]]['strike']
        df_iv_k['iv'] = iv[delta_t[i]]['sigma']
        df_iv_k.dropna(inplace=True)
        print(df_iv_k)

        # Remove outlier using curvefit
        popt, pcov = curve_fit(concave_curve, df_iv_k['K'], df_iv_k['iv'])
        errors = np.abs(df_iv_k['iv'] - concave_curve(df_iv_k['K'], *popt))
        mad = np.median(errors)
        threshold = curvefit_t * mad
        df_iv_K_out = df_iv_k[errors >= threshold]
        df_iv_k_in = df_iv_k[errors < threshold]
        print(df_iv_K_out)

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
