from GoogleNews import GoogleNews
import pandas as pd
googlenews = GoogleNews()
googlenews.enableException(True)
googlenews.set_lang('en')
googlenews.set_period('7d')
googlenews.set_encode('utf-8')
googlenews.get_news('JP Morgan')
googlenews.total_count()
print(pd.DataFrame.from_dict(googlenews.results())['datetime'].sort_value(['datetime']))
#print(googlenews.get_texts())