import numpy as np
import pandas as pd
import yfinance as yf
import yahooquery as yq
from datetime import date
import datetime
from dateutil.relativedelta import relativedelta
from matplotlib import rcParams
import matplotlib.pyplot as plt
import pypfopt as ppo
from IPython.display import display
from pypfopt.efficient_frontier import EfficientFrontier
import base64
import io

class Portfolio_Optimizer():

    def __init__(self, tickers=[], track_back = 5, rf_rate = yq.Ticker('^TNX').history(period='1y')['adjclose'][-1]/100):
        """
        Configure the optimizer
        @param tickers: The tickers included in the portfolio
        @param track_back: The amount of track back years of tickers' adjusted closed price
        @param rf_rate: risk free rate, default to be the 3 months US bill rate
        """
        self.tickers = tickers
        self.track_back = track_back
        self.rf_rate = rf_rate/100
        self.history_start_date = date.today() - relativedelta(years=track_back)
        self.data = yf.download(tickers = tickers, start = self.history_start_date.strftime('%Y-%m-%d'))['Adj Close']
        self.mean_returns = ppo.expected_returns.capm_return(self.data)
        self.cov_returns = ppo.risk_models.risk_matrix(self.data, method='ledoit_wolf')
        print("Use Portfolio_Optimizer on {} at {}".format(tickers, datetime.datetime.now()))
    
    @staticmethod
    def get_default_risk_free_rate():
        return yf.download(tickers = '^TNX')['Adj Close'][-1]
    
    def visualize_cumulative_returns(self, is_html = False):
        """
        Visualize the cumulative returns for all the tickers in the portfolio
        """
        fig = plt.figure(figsize=(12, 7))
        rcParams.update({'font.size': 22})
        plt.plot((1 + self.data.pct_change()).cumprod() - 1, label = self.data.columns)
        plt.ylabel('Cumulative Returns')
        plt.xlabel('Date')
        plt.legend(prop = {'size': 20})
        if not is_html:
            plt.show()
        else:
            img=io.BytesIO()
            plt.savefig(img,format='png')
            img.seek(0)
            plot_url = base64.b64encode(img.getvalue()).decode()
            return plot_url

    def get_mean_cov_returns(self):
        """
        Return the mean and convirance of returns in portfolio
        """
        # # print("Mean returns of each ticker")
        m_return = pd.DataFrame(self.mean_returns).T
        m_return.index = ["Return"]
        # display(m_return)
        # # print("Risk Matrix")
        # display(self.cov_returns)
        return {"mean returns": m_return, "returns covariance": self.cov_returns}
    
    def get_weight(self):
        """
        Return the weights for minimum volatility and maximum sharpe ratio
        """
        ef_sr = EfficientFrontier(self.mean_returns, self.cov_returns)
        ef_sr.max_sharpe(risk_free_rate = self.rf_rate)
        cleaned_weights_max_sharpe = ef_sr.clean_weights()
        df_cleaned_weights_max_sharpe = pd.DataFrame(cleaned_weights_max_sharpe, columns=cleaned_weights_max_sharpe.keys(), index=["Weight"])
        # print("Weight for best sharpe ratio based on data from {}".format(self.history_start_date))
        # display(df_cleaned_weights_max_sharpe)
        perf_max_sharpe = ef_sr.portfolio_performance(verbose=True, risk_free_rate = self.rf_rate)
        # print("\n")

        ef_mv = EfficientFrontier(self.mean_returns, self.cov_returns)
        ef_mv.min_volatility()
        cleaned_weights_min_vol = ef_mv.clean_weights()
        df_cleaned_weights_min_vol = pd.DataFrame(cleaned_weights_min_vol, columns=cleaned_weights_min_vol.keys(), index=["Weight"])
        # print("Weight for minimum volatility based on data from {}".format(self.history_start_date))
        # # print(cleaned_weights_min_vol.keys())
        # display(df_cleaned_weights_min_vol)
        perf_min_vol = ef_mv.portfolio_performance(verbose=True, risk_free_rate = self.rf_rate)

        return {"Best Sharpe Ratio": {"Weights": df_cleaned_weights_max_sharpe, 
                                      "Performace": {"Expected annual return": perf_max_sharpe[0],
                                                     "Annual volatility": perf_max_sharpe[1],
                                                     "Sharpe Ratio": perf_max_sharpe[2]}},
                "Minimum Volatility": {"Weights": df_cleaned_weights_min_vol, 
                                       "Performace": {"Expected annual return": perf_min_vol[0],
                                                     "Annual volatility": perf_min_vol[1],
                                                     "Sharpe Ratio": perf_min_vol[2]}}}
    
    def get_var(self, df_weights, tag = ''):
        """
        Return value at risk for given weight
        """
        a = self.data
        df = pd.DataFrame(a.values.dot(df_weights.T)).pct_change().dropna()
        df_temp = pd.DataFrame(df.values, columns = ["Return"])
        VaR_90 = df.quantile(0.1).values[0]
        VaR_95 = df.quantile(0.05).values[0]
        VaR_99 = df.quantile(0.01).values[0]
        ave_90 = df_temp[df_temp["Return"] <= VaR_90].mean().values[0]
        ave_95 = df_temp[df_temp["Return"] <= VaR_95].mean().values[0]
        ave_99 = df_temp[df_temp["Return"] <= VaR_99].mean().values[0]
        fig = plt.figure(figsize=(12, 7))
        plt.hist(df, bins=40)
        plt.axvline(x=VaR_90, label = f'90% {ave_90*100.:2f}%', linewidth = 2, linestyle = ':', color = 'red')
        plt.axvline(x=VaR_95, label = f'95% {ave_95*100.:2f}%', linewidth = 2, linestyle = '--', color = 'red')
        plt.axvline(x=VaR_99, label = f'99% {ave_99*100.:2f}%', linewidth = 2, linestyle = '-', color = 'red')
        plt.xlabel('Returns')
        plt.ylabel('Frequency')
        plt.title(label = f"Daily VaR of {tag}")
        plt.grid(True)
        plt.legend()
        img=io.BytesIO()
        plt.savefig(img,format='png')
        img.seek(0)
        plot_url_1 = base64.b64encode(img.getvalue()).decode()

        df = pd.DataFrame(a.values.dot(df_weights.T)).pct_change(periods=5).dropna()
        df_temp = pd.DataFrame(df.values, columns = ["Return"])
        VaR_90 = df.quantile(0.1).values[0]
        VaR_95 = df.quantile(0.05).values[0]
        VaR_99 = df.quantile(0.01).values[0]
        ave_90 = df_temp[df_temp["Return"] <= VaR_90].mean().values[0]
        ave_95 = df_temp[df_temp["Return"] <= VaR_95].mean().values[0]
        ave_99 = df_temp[df_temp["Return"] <= VaR_99].mean().values[0]
        fig = plt.figure(figsize=(12, 7))
        plt.hist(df, bins=40)
        plt.axvline(x=VaR_90, label = f'90% {ave_90*100.:2f}%', linewidth = 2, linestyle = ':', color = 'red')
        plt.axvline(x=VaR_95, label = f'95% {ave_95*100.:2f}%', linewidth = 2, linestyle = '--', color = 'red')
        plt.axvline(x=VaR_99, label = f'99% {ave_99*100.:2f}%', linewidth = 2, linestyle = '-', color = 'red')
        plt.xlabel('Returns')
        plt.ylabel('Frequency')
        plt.title(label = f"Weekly VaR of {tag}")
        plt.grid(True)
        plt.legend()
        img=io.BytesIO()
        plt.savefig(img,format='png')
        img.seek(0)
        plot_url_2 = base64.b64encode(img.getvalue()).decode()

        return [plot_url_1, plot_url_2]


    def random_portfolios(self, num_portfolios = 100000):
        """
        Build random portfolios and return the performance results
        """
        results = np.zeros((3,num_portfolios))
        weights_record = []
        for i in range(num_portfolios):
            weights = np.random.random(len(self.tickers))
            weights /= np.sum(weights)
            weights_record.append(weights)
            portfolio_std_dev = np.sqrt(np.dot(weights.T, np.dot(self.cov_returns, weights)))
            portfolio_return = np.sum(self.mean_returns*weights) 
            results[0,i] = portfolio_std_dev
            results[1,i] = portfolio_return
            results[2,i] = (portfolio_return - self.rf_rate) / portfolio_std_dev
        
        weight_df = pd.DataFrame(data = [weights_record[i] for i in range(len(weights_record))], columns = [i for i in self.tickers] )
        res_df = pd.DataFrame(data = [results[0], results[1], results[2]],
                              index = ['Expected Portfolio Volatility', 'Expected Portfolio Returns', 'Sharp Ratio']).T
        results_df = pd.DataFrame.join(res_df, weight_df)
        return results, weights_record, results_df

    def visualize_efficient_frontier(self, is_html = False):
        """
        This function is to visualize the efficient frontier and pointing where the target allocations are
        """
        results, weights_record, results_df = self.random_portfolios(1000)
        max_sharp_port = results_df.loc[results_df['Sharp Ratio'] == results_df['Sharp Ratio'].max()]
        min_vol_port = results_df.loc[results_df['Expected Portfolio Volatility'] == results_df['Expected Portfolio Volatility'].min()]
        min_var_loc = int(np.where(results == results[0].min())[1])
        max_sharp_loc = int(np.where(results == results[2].max())[1])
        #Visualizing our results
        fig = plt.figure(figsize=(12, 7))
        plt.scatter(x = results[0], y = results[1], c = results[2], marker='o')
        plt.xlabel('Expected volatility')
        plt.ylabel('Expected return')
        plt.colorbar(label='Sharpe ratio')
        plt.plot(max_sharp_port['Expected Portfolio Volatility'],max_sharp_port['Expected Portfolio Returns'],
                'g*', markersize=15, label = 'Portfolio with Highest Sharp Ratio')
        plt.plot(min_vol_port['Expected Portfolio Volatility'],min_vol_port['Expected Portfolio Returns'],
                'kv', markersize=15, label = 'Protfolio with Lowest Volatility')
        plt.legend()
        if not is_html:
            plt.show()
        else:
            img=io.BytesIO()
            plt.savefig(img,format='png')
            img.seek(0)
            plot_url = base64.b64encode(img.getvalue()).decode()
            return plot_url