from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from Portfolio_Optimization.Portfolio_Optimizer import Portfolio_Optimizer
from probabilistic_Valuation_Model.probabilistic_valuator import Probabilistic_Valuator
from Sentiment_plugin import sentimental_analysis
from Option_pricing_model import option_pricing

Flask_App = Flask(__name__) # Creating our Flask Instance
pd.options.display.precision = 3

def package(e):
    message = "<p style='color:Red;'><sub>Found error input, all input are reset to default.</sub></p>"
    message += "<p style='color:Red;'><sub>* Please double check your ticker input, like Apple's ticket is 'AAPL' instead of 'APPL'!</sub></p>"
    return message+f"<p style='color:Red;'><sub>* Detail: {str(e)}</sub></p>"


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
        var_max_s = po.get_var(results['Best Sharpe Ratio']["Weights"], tag = 'Best Sharpe Ratio Portfolio')
        var_min_v = po.get_var(results['Minimum Volatility']["Weights"], tag = 'Minimum Volatility Portfolio')
        sentiments = sentimental_analysis.get_setiments(tickers)

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
            sentiments_plot = sentiments[0],
            news_list = sentiments[1],
            var_max_s = var_max_s,
            var_min_v = var_min_v,
            calculation_success=True
        )
    except Exception as e:
        return render_template(
            'portfolio_optimizer.html',
            default_tickers = tickers_input,
            default_track_back_years = track_back_years_input,
            default_rfr = rf_rate_input,
            calculation_success=False,
            error= package(e)
        )

@Flask_App.route('/probailistic_valuator', methods=['GET'])
def init_probailistic_valuator():
    """ Displays the probailistic_valuator page accessible at '/probailistic_valuator' """
    print(Probabilistic_Valuator.get_default_risk_free_rate())
    return render_template('probailistic_valuator.html',
                           default_tab = 'wacc',
                           default_ticker = 'AAPL',
                           default_rfr_ticker = '^TNX',
                           default_risk_free_rate = Probabilistic_Valuator.get_default_risk_free_rate(),
                           default_market_return_rate = Probabilistic_Valuator.get_default_market_return_rate(),
                           default_t_years = 10,
                           default_beta = Probabilistic_Valuator.get_default_beta('AAPL'),
                           default_rd_reinvest = True,
                           default_intang_as_da = False,
                           default_wacc = 0.0,
                           default_wacc_std = 0.0)

@Flask_App.route('/probailistic_valuator/calculate_wacc', methods=['POST'])
def operation_wacc():
    """Route where we send results"""
    # request.form looks for:
    # html tags with matching "name= "
    ticker_input = request.form['Ticker']  
    rfr_Ticker = request.form['rfr_Ticker']
    market_return_rate = float(request.form['market_return_rate'])
    rd_reinvest = request.form.get('rd_reinvest')
    intang_as_da = request.form.get('intang_as_da')

    try:
        pv = Probabilistic_Valuator(ticker = ticker_input, risk_free_rate_ticker = rfr_Ticker, 
                                    market_return = market_return_rate, beta = None, 
                                    rd_in_reinvest = rd_reinvest, intang_as_da = intang_as_da)
        pv.get_wacc()

        return render_template(
            'probailistic_valuator.html',
            default_tab = 'wacc',
            default_ticker = pv.ticker,
            default_rfr_ticker = pv.risk_free_rate_ticker,
            default_risk_free_rate = pv.risk_free_rates[-1],
            default_market_return_rate = pv.market_return,
            default_t_years = pv.t_years,
            default_beta = pv.beta,
            default_rd_reinvest = pv.rd_in_reinvest,
            default_intang_as_da = pv.intang_as_da,
            fundamentals = pv.fundamentals.to_html(),
            fundamentals_plot = pv.fundamentals_plot,
            wacc_dist = pv.wacc[2],
            wacc_explain = pv.wacc[3],
            default_wacc = pv.wacc[0][0],
            default_wacc_std = np.std(pv.wacc[0]),
            calculation_wacc_success=True,
            simulation_success = False
        )
    
    except Exception as e:
        return render_template(
            'probailistic_valuator.html',
            default_tab = 'wacc',
            default_ticker = 'AAPL',
            default_rfr_ticker = '^TNX',
            default_risk_free_rate = Probabilistic_Valuator.get_default_risk_free_rate(),
            default_market_return_rate = Probabilistic_Valuator.get_default_market_return_rate(),
            default_t_years = 10,
            default_beta = Probabilistic_Valuator.get_default_beta('AAPL'),
            default_rd_reinvest = True,
            default_intang_as_da = False,
            default_wacc = 0.0,
            default_wacc_std = 0.0,
            calculation_wacc_success=False,
            simulation_success = False,
            error= package(e)
        )

@Flask_App.route('/probailistic_valuator/run_sim', methods=['POST'])
def operation_simulation():
    """Route where we send results"""
    ticker_input = request.form['Ticker']  
    rfr_Ticker = request.form['rfr_Ticker']
    market_return_rate = float(request.form['market_return_rate'])
    t_years = int(request.form['t_years'])
    rd_reinvest = request.form.get('rd_reinvest')
    intang_as_da = request.form.get('intang_as_da')
    try:
        pv = Probabilistic_Valuator(ticker = ticker_input, risk_free_rate_ticker = rfr_Ticker, 
                                    market_return = market_return_rate, beta = None, 
                                    rd_in_reinvest = rd_reinvest, intang_as_da = intang_as_da)
        [target_price, current_price, price_dist, FCFF_mean, pv_explain, FCFF_mean_plot] = pv.run_simulations(t_intervals = t_years)

        return render_template(
            'probailistic_valuator.html',
            default_tab = 'sim',
            default_ticker = pv.ticker,
            default_rfr_ticker = pv.risk_free_rate_ticker,
            default_risk_free_rate = pv.risk_free_rates[-1],
            default_market_return_rate = pv.market_return,
            default_t_years = pv.t_years,
            default_beta = pv.beta,
            default_rd_reinvest = pv.rd_in_reinvest,
            default_intang_as_da = pv.intang_as_da,
            fundamentals = pv.fundamentals.to_html(),
            fundamentals_plot = pv.fundamentals_plot,
            wacc_dist = pv.wacc[2],
            wacc_explain = pv.wacc[3],
            default_wacc = pv.wacc[0][0],
            default_wacc_std = np.std(pv.wacc[0]),
            calculation_wacc_success=True,
            FCFF_mean = FCFF_mean.to_html(),
            FCFF_mean_plot = FCFF_mean_plot,
            price_dist = price_dist,
            pv_explain = pv_explain,
            simulation_success = True
        )
    
    except Exception as e:
        return render_template(
            'probailistic_valuator.html',
            default_ticker = 'AAPL',
            default_rfr_ticker = '^TNX',
            default_risk_free_rate = Probabilistic_Valuator.get_default_risk_free_rate(),
            default_market_return_rate = Probabilistic_Valuator.get_default_market_return_rate(),
            default_t_years = 10,
            default_beta = Probabilistic_Valuator.get_default_beta('AAPL'),
            default_rd_reinvest = True,
            default_intang_as_da = False,
            default_wacc = 0.0,
            default_wacc_std = 0.0,
            default_tab = 'sim',
            simulation_success=False,
            calculation_wacc_success=False,
            error= package(e)
        )

@Flask_App.route('/option_analysis', methods=['GET'])
def init_option_analysis():
    """ Displays the option_analysis page accessible at '/option_analysis' """

    return render_template('option_analysis.html',
                           default_ticker = "AAPL",
                           default_option_type = "call",
                           default_expiry_weeks = 5,
                           default_rfr = Portfolio_Optimizer.get_default_risk_free_rate()/100,
                           default_trade_date = 3,
                           default_curvefit_t = 3)

@Flask_App.route('/option_analysis/result', methods=['POST'])
def option_result():
    """Route where we send results"""
    ticker_input = request.form['Ticker']
    option_type = request.form['option_type']
    expiry_weeks = int(request.form['expiry'])
    rf_rate = float(request.form['rf_rate'])
    trade_date = int(request.form['trade_date'])
    curvefit_t = int(request.form['curvefit_t'])
    
    try:
        [plot, option_table] = option_pricing.get_iv_plot(sym = ticker_input, weeks = expiry_weeks, option_type = option_type,
                                                           rf= rf_rate, trade_date = trade_date, curvefit_t = curvefit_t)

        return render_template(
            'option_analysis.html',
            default_ticker = ticker_input,
            default_option_type = option_type,
            default_expiry_weeks = expiry_weeks,
            default_rfr = rf_rate,
            default_trade_date = trade_date,
            default_curvefit_t = curvefit_t,
            plot_url = plot,
            option_table_keys = list(option_table.keys()),
            option_table = option_table,
            calculation_success = True
        )
    
    except Exception as e:
        return render_template(
            'option_analysis.html',
            default_ticker = ticker_input,
            default_option_type = option_type,
            default_expiry_weeks = expiry_weeks,
            default_rfr = rf_rate,
            default_trade_date = trade_date,
            default_curvefit_t = curvefit_t,
            calculation_success = False,
            error= package(e)
        )

if __name__ == '__main__':
    Flask_App.debug = True
    Flask_App.run(host='0.0.0.0', port=80)