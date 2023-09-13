import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
import pandas as pd
from urllib.request import urlopen, Request
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
from datetime import datetime
from datetime import date
from dateutil.relativedelta import relativedelta
import base64
import io
nltk.download('vader_lexicon')

def get_setiments(tickers = ['NFLX','PTON','IDXX','HSY','PYPL']):
    
    print("Use get_setiments on {} at {}".format(tickers, datetime.now()))
    web_url = 'https://finviz.com/quote.ashx?t='
    news_tables = {}
    for tick in tickers:
        url = web_url + tick
        try:
            req = Request(url=url,headers={"User-Agent": "Chrome Mozilla/5.0 (Macintosh; Intel Mac OS X 10.9; rv:32.0) Gecko/20100101 Firefox/32.0"}) 
            response = urlopen(req)    
            html = BeautifulSoup(response,"lxml")
            news_table = html.find(id='news-table')
            news_tables[tick] = news_table
        except Exception as e:
            print("Failed to get news on {}".format(tick))

    if not news_tables:
        return [None, [None, None]]

    news_list = []
    for file_name, news_table in news_tables.items():
        for i in news_table.findAll('tr'):
            if i.a is None or i.td is None:
                continue
            text = i.a.get_text() 
            
            date_scrape = i.td.text.split()

            if len(date_scrape) == 1:
                time = date_scrape[0]
            else:
                if date_scrape[0] == 'Today':
                    dt = date.today()
                else:
                    dt = datetime.strptime(date_scrape[0], '%b-%d-%y').date()
                time = date_scrape[1]

            tick = file_name.split('_')[0]
            if dt >= date.today() - relativedelta(days=4):
                news_list.append([tick, dt.strftime('%b-%d'), time, text])

    vader = SentimentIntensityAnalyzer()
    columns = ['ticker', 'date', 'time', 'headline']
    news_df = pd.DataFrame(news_list, columns=columns)
    scores = news_df['headline'].apply(vader.polarity_scores).tolist()
    scores_df = pd.DataFrame(scores)
    news_df = news_df.join(scores_df, rsuffix='_right')
    # print(news_df.head())
    mean_scores = news_df.groupby(['ticker','date']).mean()
    mean_scores = mean_scores.unstack()

    mean_scores = mean_scores.xs('compound', axis="columns").transpose().fillna(0)
    mean_scores.loc['Average'] = mean_scores.mean()
    # print(mean_scores)

    fig = mean_scores.plot(kind = 'bar', figsize=(12, 7), rot=0)
    plt.rcParams["figure.autolayout"] = True
    plt.legend(bbox_to_anchor=(1.0, 1.0))
    plt.grid()
    img=io.BytesIO()
    plt.savefig(img,format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    return [plot_url, [web_url, tickers]]