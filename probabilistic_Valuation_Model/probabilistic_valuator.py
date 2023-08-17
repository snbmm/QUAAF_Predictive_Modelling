import yahooquery as yq
import yfinance as yf
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import base64
import io
from dateutil.relativedelta import relativedelta

class Probabilistic_Valuator():
    def __init__(self, ticker, get_estimation = True, freq = 'a', 
                 risk_free_rate_ticker = None, market_return = None, 
                 beta = None, rd_in_reinvest = True, intang_as_da = False):
                 
        self.ticker = ticker
        self.get_estimation = get_estimation
        self.risk_free_rate_ticker = risk_free_rate_ticker
        self.t_years = 10
        if risk_free_rate_ticker:
            self.risk_free_rates = yq.Ticker(risk_free_rate_ticker).history(period='1y')['adjclose']/100
        else:
            self.risk_free_rates = yq.Ticker('^TNX').history(period='1y')['adjclose']/100
            self.risk_free_rate_ticker = '^TNX'
        sp500 = yq.Ticker('^GSPC').history(period='30y')
        self.market_return = market_return or (sp500['adjclose'][-1] / sp500['adjclose'][0]) ** (252/len(sp500)) - 1
        self.freq = 'a'
        yq_ticker = yq.Ticker(ticker)
        self.df_is = yq_ticker.income_statement(frequency=freq, trailing = False).set_index('asOfDate')
        self.df_bs = yq_ticker.balance_sheet(frequency=freq, trailing = False).set_index('asOfDate')
        self.df_cf = yq_ticker.cash_flow(frequency=freq, trailing = False).set_index('asOfDate')
        self.df_history = yq_ticker.history(start = self.df_bs.index[0])['adjclose']
        self.df_key_stats = yf.Ticker(ticker).info
        self.beta = beta or self.df_key_stats['beta']
        self.rd_in_reinvest = rd_in_reinvest
        self.intang_as_da = intang_as_da
        [self.fundamentals, self.fundamentals_plot] = self.get_fundamentals(intang_as_da = self.intang_as_da)
        self.wacc = None
        print("Use Probabilistic_Valuator on {} at {}".format(ticker, datetime.datetime.now()))
    
    @staticmethod
    def get_default_beta(ticker = 'AAPL'):
        try:
            return yq.Ticker(ticker).key_stats[ticker]['beta']
        except Exception as e:
            return yf.Ticker(ticker).info['beta']
    
    @staticmethod
    def get_default_risk_free_rate(ticker = '^TNX'):
        rfr = yq.Ticker(ticker).history(period='1y')['adjclose'][-1]/100
        if isinstance(rfr, float):
            return rfr
        else:
            return yf.download(tickers = ticker)['Adj Close'][-1]/100
    
    @staticmethod
    def get_default_market_return_rate():
        sp500 = yq.Ticker('^GSPC').history(period='30y')
        return (sp500['adjclose'][-1] / sp500['adjclose'][0]) ** (252/len(sp500)) - 1
    
    def get_fundamentals(self, intang_as_da = False):
        revenue = self.df_is['TotalRevenue'].fillna(0)
        Chg_WC = self.df_cf['ChangeInWorkingCapital'].fillna(0)
        eff_tax = self.df_is['TaxRateForCalcs'].fillna(0)
        capex = abs(self.df_cf['CapitalExpenditure'].fillna(0))
        depreciation = self.df_cf['DepreciationAndAmortization'].fillna(0)
        if ('AmortizationOfIntangibles' in self.df_cf) and not intang_as_da:
            depreciation -= self.df_cf['AmortizationOfIntangibles'].fillna(0)
        if ('ResearchAndDevelopment' in self.df_is):
            research_d = abs(self.df_is['ResearchAndDevelopment'].fillna(0))
        else:
            research_d = capex - capex
        ebit = self.df_is['EBIT']
        after_tax_EBIT = ebit * (1 - eff_tax)
        if self.rd_in_reinvest:
            reinv_rate = (capex + research_d - depreciation + Chg_WC) / after_tax_EBIT
        else:
            reinv_rate = (capex - depreciation + Chg_WC) / after_tax_EBIT
        tic = self.df_bs['TotalAssets'].fillna(0) - self.df_bs['PayablesAndAccruedExpenses'].fillna(0) - (
            self.df_bs['CashCashEquivalentsAndShortTermInvestments'].fillna(0) - (
            self.df_bs['CurrentLiabilities'].fillna(0)- self.df_bs['CurrentAssets'].fillna(0) + self.df_bs['CashCashEquivalentsAndShortTermInvestments'].fillna(0)
            ).clip(lower=0)
        )
        roc = after_tax_EBIT / tic
        g = (reinv_rate * roc)
        fundamentals = pd.concat([revenue, Chg_WC, eff_tax, capex, research_d, depreciation, ebit, after_tax_EBIT,
                                        reinv_rate, roc, g], axis=1)
        fundamentals.columns = ['Revenue','Change in Working Capital',
                                        'Effective Tax Rate', 'Capex', 'R&D', 'Depr. & Amort.', 'EBIT','EBIT (1-t)',
                                        'Reinvestment Rate', 'ROC', 'g']
        fundamentals.index.name = None
        fundamentals.index = fundamentals.index.strftime('%Y-%m-%d')
        
        
        #fig = fundamentals.plot(kind='bar',rot=0, colormap='jet',figsize=(12,7), subplots=True, layout=(4,3), legend = False, grid = True)
        figure, axis = plt.subplots(1, 2, figsize=(12,4))
        figure.tight_layout()
        #axis[0] = fundamentals[['Revenue','Change in Working Capital', 'Capex', 'R&D', 'Depr. & Amort.', 'EBIT','EBIT (1-t)']].plot.bar(rot=0)
        fundamentals.plot(kind='bar', y = ['Revenue','Change in Working Capital', 'Capex', 'R&D', 'Depr. & Amort.', 'EBIT','EBIT (1-t)'], ax=axis[0], rot = 0)
        axis[0].set_yscale('log')
        axis[0].legend(ncol=2)
        axis[0].grid(True)
        #axis[1] = fundamentals[['Effective Tax Rate', 'Reinvestment Rate', 'ROC', 'g']].plot.bar(rot=0)

        fundamentals.plot(kind='bar', y = ['Effective Tax Rate', 'Reinvestment Rate', 'ROC', 'g'], ax=axis[1], rot = 0)
        axis[1].legend(ncol=2)
        axis[1].grid(True)

        img=io.BytesIO()
        plt.savefig(img,format='png')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()

        return [fundamentals, plot_url]
    
    def get_wacc(self, iteration = 5000, market_return = None, beta = None):
        """
        Calculate WACC using yahoo finance data, through yahooquery package (yfinance is not stable to get the sheets)
        @param tickers:         The tickers we want to calculate WACC
        @param get_estimation:  Do we want to include the estimation results instead of sigle WACC value
        @param n:               Sample amount
        @param freq:            Using last 4 annual or quater information, choose 'a' as annual if the company does not reveal quater report
        @param risk_free_rate:  Using default ^TNX rate as rish free rate, or we can set our own
        @param market_return:   Using default sp500 30 years average annual return rate as market return, or we can set our own
        @param beta:            Using beta from yahoo finance or we can set our own
        @return:                Return a list of WACC, the first one is the calculation using latest data, the rest is the estimation, the len
                                should be n+1, return rates used to calculate the result.
        """
        # Get the risk-free rate
        risk_free_rate = self.risk_free_rates[-1]
        rfr_dist = [self.risk_free_rates.mean(), self.risk_free_rates.std()]

        # print("risk_free_rate: {}".format(risk_free_rate))

        # Get the market return
        market_return = self.market_return
        # print("market_return: {}".format(market_return))

        # Get the beta
        beta = self.beta
        # print("beta: {}".format(beta))

        # Calculate the cost of equity using the CAPM
        cost_of_equity = risk_free_rate + beta * (market_return - risk_free_rate)
        # print("cost_of_equity: {}".format(cost_of_equity))

        # Form a table for all the info needed to calculate WACC
        bs_list = ['periodType', 'TotalDebt']
        is_list = ['InterestExpense', 'TaxRateForCalcs']
        def_values = {'periodType': '3M' if self.freq == 'q' else '12M',
                    'TotalDebt': 0,
                    'InterestExpense': 0,
                    'TaxRateForCalcs': 0}

        for k in bs_list:
            if k not in self.df_bs:
                self.df_bs[k] = def_values[k]
                print("-Warning- Missing column in balance sheet, set default value: {} in column {}".format(def_values[k], k))

        for k in is_list:
            if k not in self.df_is:
                self.df_is[k] = def_values[k]
                print("-Warning- Missing column in income statement sheet, set default value: {} in column {}".format(def_values[k], k))

        info = self.df_bs[bs_list].join(self.df_is[is_list], on = 'asOfDate')
        if info.isnull().values.any():
            info = info.fillna(0)
            print("-Warning- Found missing value, filled with 0")
        info['interest_rate'] = (info['InterestExpense']/info['TotalDebt']+1)** (4 if self.freq == 'q' else 1 ) -1

        #Calculate latest WACC

        # Get the interest rate on Microsoft's debt
        interest_rate = info['interest_rate'][-1]
        # print("interest_rate: {}".format(interest_rate))

        # Get the tax rate
        tax_rate = info['TaxRateForCalcs'][-1]
        # print("tax_rate: {}".format(tax_rate))

        # Calculate the cost of debt
        cost_of_debt = interest_rate * (1 - tax_rate)
        # print("cost_of_debt: {}".format(cost_of_debt))

        # Get the market capitalization
        market_cap = self.df_key_stats['marketCap']
        # print("market_cap: {}".format(market_cap))

        # Get the total debt
        total_debt = info['TotalDebt'][-1]
        # print("total_debt: {}".format(total_debt))

        # Calculate the weights of equity and debt in the capital structure
        equity_weight = market_cap / (market_cap + total_debt)
        # print("equity_weight: {}".format(equity_weight))
        debt_weight = total_debt / (market_cap + total_debt)
        # print("debt_weight: {}".format(debt_weight))

        # Calculate the WACC
        wacc = equity_weight * cost_of_equity + debt_weight * cost_of_debt
        # print("wacc: {}".format(wacc))

        # # print the results
        # print(f'WACC for {self.ticker}: {wacc:.2%}')
        ## print(f'Standard deviation of WACC for {ticker}: {wacc_std:.2%}')

        # WACC distribution
        wacc_list = [wacc]
        rates_list = [[risk_free_rate, interest_rate, tax_rate]]
        i = 0
        for i in range(iteration):
            risk_free_rate_r = np.random.normal(rfr_dist[0], rfr_dist[1])
            interest_rate_r = np.random.normal(np.mean(info['interest_rate']), np.std(info['interest_rate']))
            tax_rate_r = np.random.normal(np.mean(info['TaxRateForCalcs']), np.std(info['TaxRateForCalcs']))
            cost_of_equity_r = risk_free_rate_r + beta * (market_return - risk_free_rate_r)
            cost_of_debt_v = interest_rate_r * (1 - tax_rate_r)
            wacc_list.append(equity_weight * cost_of_equity_r + debt_weight * cost_of_debt_v)
            rates_list.append([risk_free_rate_r, interest_rate, tax_rate])
        fig = plt.figure(figsize=(6, 4))
        fig.tight_layout()
        plt.rcParams["figure.autolayout"] = True
        plt.xlabel(f'Estimated {self.ticker} WACC')
        plt.ylabel('Counts')
        plt.axvline(x=wacc_list[0], label = f'Current WACC {round(wacc_list[0],6)}', linewidth = 3, color = 'gray')
        plt.axvline(x=np.median(wacc_list), label = f'Median WACC {round(np.mean(wacc_list),6)}',
                    linewidth = 3, color = 'purple')

        plt.hist(wacc_list, bins = 100)
        #sns.histplot(wacc_list, color='orange', label = 'Simulations').set_xticklabels(xlabels)
        plt.legend()
        img=io.BytesIO()
        plt.savefig(img,format='png')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()
        explain =   f"Cost of equity:\n cost_of_equity({cost_of_equity:.2f}) \n = risk_free_rate({risk_free_rate:.2f}) + \
                    beta({beta:.2f}) * (market_return({market_return:.2f}) \n- risk_free_rate({risk_free_rate:.2f}))\n\n \
                    Latest Interest Rate:\n Interest Rate({info['interest_rate'][-1]:.2f}) \n = \
                    InterestExpense({info['InterestExpense'][-1]:.2e}) / TotalDebt({info['TotalDebt'][-1]:.2e})\n\n\
                    Cost of Debt:\n cost_of_debt({cost_of_debt:.2f}) \n = \
                    interest_rate({interest_rate:.2f}) * (1 - tax_rate({tax_rate:.2f}))\n\n\
                    Equity Weight:\n equity_weight({equity_weight:.2f}) \n = \
                    market_cap({market_cap:.2e}) \n/ (market_cap({market_cap:.2e}) + total_debt({total_debt:.2e}))\n\n\
                    WACC:\n wacc({wacc:.4f}) \n = equity_weight({equity_weight:.2f}) * \
                    cost_of_equity({cost_of_equity:.2f}) \n+ debt_weight({debt_weight:.2f}) * cost_of_debt({cost_of_debt:.2f})\n"
        
        self.wacc = [wacc_list, rates_list, plot_url, explain.replace("\n", "<br />")]
        return [wacc_list, rates_list, plot_url, explain]

    def get_ev_ebit_ratios(self):
        ev_ebit_ratio = []
        price = self.df_history
        print(price.index)
        price.index = price.index.droplevel(0).map(str)
        for i in range(len(self.df_bs.index) -2, len(self.df_bs.index)-1):
            start = self.df_bs.index[i]
            if i+1 == len(self.df_bs.columns):
                end = date.today()
            else:
                end = self.df_bs.index[i+1] 
            for date in pd.date_range(start = start, end = end, freq = '1d').tolist():
                if date.strftime('%Y-%m-%d') in price.index:
                    ev = price[date.strftime('%Y-%m-%d')] * self.df_key_stats['sharesOutstanding'] + self.df_bs['TotalDebt'][i] - self.df_bs['CashAndCashEquivalents'][i]
                    ev_ebit_ratio.append(ev/self.df_is['EBIT'][i])
        return [np.mean(ev_ebit_ratio), np.std(ev_ebit_ratio)]
    
    def run_simulations(self, iteration = 5000, t_intervals = 10, wacc = None):
        self.t_years = t_intervals
        values = []
        sharesOutstanding = self.df_key_stats['sharesOutstanding']
        def_ev_ebit_ratios = self.get_ev_ebit_ratios()
        if wacc is None:
            if self.wacc:
                wacc = self.wacc[0]
            wacc = self.get_wacc()[0]
        future_date = [(self.df_is.index[-1].date() + relativedelta(years=i+1)).strftime('%Y-%m-%d') for i in range(t_intervals)]
        df_result = pd.DataFrame()
        p_termial_v = 0
        TerMulSim_mean = 0
        if 'NetDebt' in self.df_bs and not np.isnan(self.df_bs['NetDebt'][-1]):
            # print(self.df_bs['NetDebt'][-1])
            net_debt = self.df_bs['NetDebt'][-1]
        else:
            net_debt = self.df_bs['TotalDebt'][-1] - self.df_bs['CashAndCashEquivalents'][-1]

        for i in range(iteration):
            Expected_g = np.random.normal(np.mean(self.fundamentals['g']), np.std(self.fundamentals['g']), size = t_intervals)
            simWACC = np.random.normal(wacc[0], np.std(wacc), size= t_intervals)
            Reinvestment_rate = np.random.normal(np.mean(self.fundamentals['Reinvestment Rate']), np.std(self.fundamentals['Reinvestment Rate']), size = t_intervals)
            Reinvestment_rate = np.clip(Reinvestment_rate, -0.5, 0.5)
            tax = np.random.normal(np.mean(self.fundamentals['Effective Tax Rate']), np.std(self.fundamentals['Effective Tax Rate']), size = t_intervals)
            change_in_working_capital = np.random.normal(np.mean(self.fundamentals['Change in Working Capital']), np.std(self.fundamentals['Change in Working Capital']), size = t_intervals)
            TerMulSim = np.random.normal(def_ev_ebit_ratios[0], def_ev_ebit_ratios[1])
            TerMulSim = min(TerMulSim, 50)
            TerMulSim_mean = (TerMulSim + i*TerMulSim_mean)/(i+1)

            EBIT_E = []
            discount_factor = []
            a = self.df_is['EBIT'][-1]
            for n in range(t_intervals):
                a *= (1 + Expected_g[n])
                EBIT_E.append(a)
                if n == 0:
                    discount_factor.append(1 + simWACC[n])
                else:
                    new_discount_f = discount_factor[-1]*(1 + simWACC[n])
                    discount_factor.append(new_discount_f)

            after_tax_EBIT = (EBIT_E * (1 - tax))

            Capex_Dep = (after_tax_EBIT * Reinvestment_rate) - change_in_working_capital

            FCFF = after_tax_EBIT - (after_tax_EBIT * Reinvestment_rate)
            if i == 0:
                df_result = pd.DataFrame([EBIT_E, Reinvestment_rate, discount_factor, FCFF], 
                                         columns = future_date,
                                         index = ['EBIT', 'Reinvestment rate', 'Discount Rate', 'FCFF'])
            else:
                df_result = (1/(i+1))*pd.DataFrame([EBIT_E, Reinvestment_rate, discount_factor, FCFF], 
                                                   columns = future_date,
                                                   index = ['EBIT', 'Reinvestment rate', 'Discount Rate', 'FCFF']) + (i/(i+1))*df_result

            PV = FCFF / discount_factor
            terminalValue = (EBIT_E[-1] * TerMulSim)
            PV_tV = terminalValue / ((1 + wacc[0]) ** t_intervals)
            p_termial_v = PV_tV*1/(i+1) + p_termial_v*i/(i+1)
            equityValue = PV.sum() + PV_tV - net_debt
            v = equityValue / sharesOutstanding
            values.append(v)
        # print(Expected_g)
        # print(net_debt)
        # print(PV_tV)
        # print(PV.sum())
        # print(equityValue)
        # print(sharesOutstanding)

        fig_df_result = df_result.T.plot(kind='bar',rot=45, colormap='jet',figsize=(12,6), subplots=True, layout=(2,2), legend = False, grid = True)
        plt.rcParams["figure.autolayout"] = True
        plt.grid()
        plt.tight_layout()
        img=io.BytesIO()
        plt.savefig(img,format='png')
        img.seek(0)
        plot_url_1 = base64.b64encode(img.getvalue()).decode()

        fig = plt.figure(figsize=(6, 4))
        plt.rcParams["figure.autolayout"] = True
        plt.xlabel(f'Estimated {self.ticker} Price $')
        plt.ylabel('Counts')
        plt.axvline(x=self.df_history[-1], label = f'Current Price ${self.df_history[-1]}', linewidth = 3, color = 'gray')
        plt.axvline(x=np.median(values), label = f'Median Value ${round(np.mean(values),2)}',
                    linewidth = 3, color = 'purple')

        plt.hist(values, bins = 100)
        #sns.histplot(wacc_list, color='orange', label = 'Simulations').set_xticklabels(xlabels)
        plt.legend()
        img=io.BytesIO()
        plt.savefig(img,format='png')
        img.seek(0)
        plot_url_2 = base64.b64encode(img.getvalue()).decode()
        explain = f"WACC / Discount rate: {wacc[0]:.2f} \n Estimated EV/EBIT: {TerMulSim_mean:.2f}\n\n\
            Estimated Stock Value\n\
                 =  (Present value of FCF ({PV.sum():.2e}) \n\
                    + Estimated EV/EBIT ({def_ev_ebit_ratios[0]:.2f}) * Terminal EBIT ({EBIT_E[-1]:.2e}) \n\
                        - NetDebt ({net_debt:.2e})) \n\
                            / shares Outstanding ({sharesOutstanding:.2e})\n\
                                = {np.mean(values):.2f}\n\n"
        nf_other_model = False
        if np.mean(self.fundamentals['Reinvestment Rate']) > 0.5 or np.mean(self.fundamentals['Reinvestment Rate']) < -0.5:
            nf_other_model =True
            explain += f"<p style='color:Red;'><sub>* The reinvestment rate mean {np.mean(self.fundamentals['Reinvestment Rate']):.2f} is clipped at range [-0.5, 0.5]</sub></p>"
        if def_ev_ebit_ratios[0].mean() > 50:
            nf_other_model =True
            explain += f"<p style='color:Red;'><sub>* The EV/EBIT mean {def_ev_ebit_ratios[0].mean()} is capped at 50X </sub></p>"
        if nf_other_model:
            explain += f"<p style='color:Red;'><sub>Please consider to use another model!</sub></p>"
        return [round(np.mean(values),2), np.mean(values)/self.df_history[-1] -1, plot_url_2, df_result, explain.replace("\n", "<br />"), plot_url_1]

#a= Probabilistic_Valuator(ticker = 'AAPL')
#a.run_simulations()
