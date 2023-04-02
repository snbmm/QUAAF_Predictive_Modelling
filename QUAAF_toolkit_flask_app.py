from flask import Flask, render_template, request
import pandas as pd
from Portfolio_Optimization.Portfolio_Optimizer import Portfolio_Optimizer

Flask_App = Flask(__name__) # Creating our Flask Instance

@Flask_App.route('/', methods=['GET'])
def main():
    """ Displays the entrance page accessible at '/' """

    return render_template('main_page.html')

@Flask_App.route('/portfolio_optimizer', methods=['GET'])
def init_portfolio_optimizer():
    """ Displays the portfolio_optimizer page accessible at '/portfolio_optimizer' """

    return render_template('portfolio_optimizer.html',
                           default_tickers = "NFLX,DOCU,IDXX,HSY,PYPL",
                           default_track_back_years = 5,
                           default_rfr = Portfolio_Optimizer.get_default_risk_free_rate())

@Flask_App.route('/portfolio_optimizer', methods=['POST'])
def operation_result():
    """Route where we send results"""

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
            po = Portfolio_Optimizer(tickers, track_back_years, rf_rate)
        else:
            po = Portfolio_Optimizer(tickers, track_back_years)
        returns = po.get_mean_cov_returns()
        results = po.get_weight()

        return render_template(
            'portfolio_optimizer.html',
            default_tickers = tickers_input,
            default_track_back_years = track_back_years_input,
            default_rfr = rf_rate_input,
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
            'portfolio_optimizer.html',
            calculation_success=False,
            error=e
        )

if __name__ == '__main__':
    Flask_App.debug = True
    Flask_App.run(host='0.0.0.0', port=80)