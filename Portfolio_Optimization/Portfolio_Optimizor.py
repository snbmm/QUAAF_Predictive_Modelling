import numpy as np
import pandas as pd
import yfinance as yf
from datetime import date
from dateutil.relativedelta import relativedelta
from matplotlib import rcParams
import matplotlib.pyplot as plt
import pypfopt as ppo
from IPython.display import display
from pypfopt.efficient_frontier import EfficientFrontier
import base64
import io

class Portfolio_Optimizor():

    def __init__(self, tickers=[], track_back = 5, rf_rate = yf.download(tickers = '^IRX')['Adj Close'][-1]):
        """
        Configure the optimizor
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
    
    def visualize_cumulative_returns(self, is_html = False):
        """
        Visualize the cumulative returns for all the tickers in the portfolio
        """
        fig = plt.figure(figsize=(20, 10))
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
        print("Mean returns of each ticker")
        m_return = pd.DataFrame(self.mean_returns).T
        m_return.index = ["Return"]
        display(m_return)
        print("Risk Matrix")
        display(self.cov_returns)
        return {"mean returns": m_return, "returns covariance": self.cov_returns}
    
    def get_weight(self):
        """
        Return the weights for minimum volatility and maximum sharpe ratio
        """
        ef_sr = EfficientFrontier(self.mean_returns, self.cov_returns)
        ef_sr.max_sharpe(risk_free_rate = self.rf_rate)
        cleaned_weights_max_sharpe = ef_sr.clean_weights()
        df_cleaned_weights_max_sharpe = pd.DataFrame(cleaned_weights_max_sharpe, columns=cleaned_weights_max_sharpe.keys(), index=["Weight"])
        print("Weight for best sharpe ratio based on data from {}".format(self.history_start_date))
        display(df_cleaned_weights_max_sharpe)
        perf_max_sharpe = ef_sr.portfolio_performance(verbose=True, risk_free_rate = self.rf_rate)
        print("\n")

        ef_mv = EfficientFrontier(self.mean_returns, self.cov_returns)
        ef_mv.min_volatility()
        cleaned_weights_min_vol = ef_mv.clean_weights()
        df_cleaned_weights_min_vol = pd.DataFrame(cleaned_weights_min_vol, columns=cleaned_weights_min_vol.keys(), index=["Weight"])
        print("Weight for minimum volatility based on data from {}".format(self.history_start_date))
        print(cleaned_weights_min_vol.keys())
        display(df_cleaned_weights_min_vol)
        perf_min_vol = ef_mv.portfolio_performance(verbose=True, risk_free_rate = self.rf_rate)

        return {"Best Sharpe Ratio": {"Weights": df_cleaned_weights_max_sharpe, 
                                      "Performace": {"Expected annual return": perf_max_sharpe[0],
                                                     "Annual volatility": perf_max_sharpe[1],
                                                     "Sharpe Ratio": perf_max_sharpe[2]}},
                "Minimum Volatility": {"Weights": df_cleaned_weights_min_vol, 
                                       "Performace": {"Expected annual return": perf_min_vol[0],
                                                     "Annual volatility": perf_min_vol[1],
                                                     "Sharpe Ratio": perf_min_vol[2]}}}
    
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
        results, weights_record, results_df = self.random_portfolios(100000)
        max_sharp_port = results_df.loc[results_df['Sharp Ratio'] == results_df['Sharp Ratio'].max()]
        min_vol_port = results_df.loc[results_df['Expected Portfolio Volatility'] == results_df['Expected Portfolio Volatility'].min()]
        min_var_loc = int(np.where(results == results[0].min())[1])
        max_sharp_loc = int(np.where(results == results[2].max())[1])
        #Visualizing our results
        fig = plt.figure(figsize=(20, 10))
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


from flask import Flask, render_template, request

Flask_App = Flask(__name__) # Creating our Flask Instance

@Flask_App.route('/', methods=['GET'])
def index():
    """ Displays the index page accessible at '/' """

    return render_template('index.html')

@Flask_App.route('/operation_result/', methods=['POST'])
def operation_result():
    """Route where we send calculator form input"""

    error = None
    result = None

    # request.form looks for:
    # html tags with matching "name= "
    tickers_input = request.form['Tickers']  
    track_back_years_input = request.form['Track_back_years']
    rf_rate_input = request.form['rf_rate']

    try:
        tickers = tickers_input.replace(';',' ').replace(',',' ').replace("'",' ').split()
        track_back_years = int(track_back_years_input)
        if rf_rate_input:
            rf_rate = float(rf_rate_input)
            po = Portfolio_Optimizor(tickers, track_back_years, rf_rate)
        else:
            po = Portfolio_Optimizor(tickers, track_back_years)
        returns = po.get_mean_cov_returns()
        results = po.get_weight()

        return render_template(
            'index.html',
            mean_return = returns['mean returns'].to_html(),
            cov_returns = returns['returns covariance'].to_html(),
            cumulative_returns = po.visualize_cumulative_returns(is_html = True),
            efficient_frontier = po.visualize_efficient_frontier(is_html = True),
            bsr_weight = results["Best Sharpe Ratio"]["Weights"].to_html(),
            bsr_perf = pd.DataFrame(results["Best Sharpe Ratio"]["Performace"], index=[0]).to_html(index=False),
            mv_weight = results["Minimum Volatility"]["Weights"].to_html(),
            mv_perf = pd.DataFrame(results["Minimum Volatility"]["Performace"], index=[0]).to_html(index=False),
            calculation_success=True
        )
        
    except ValueError as e:
        return render_template(
            'index.html',
            calculation_success=False,
            error=e
        )

if __name__ == '__main__':
    Flask_App.debug = True
    Flask_App.run()
