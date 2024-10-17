import pandas as pd

global lost_percentage_csrt
global total_time_crst
global recovery_rate_crst
global lost_percentage_median
global total_time_median
global recovery_rate_median
global lost_percentage_mosse
global total_time_mosse
global recovery_rate_mosse

lost_percentage_csrt = []
total_time_crst = []
recovery_rate_crst = []
lost_percentage_median = []
total_time_median = []
recovery_rate_median = []
lost_percentage_mosse = []
total_time_mosse = []
recovery_rate_mosse = []

def sumTable():
    results = {
        'Method': ['MedainFlow', 'CSRT', 'MOSSE'],
        'Частота потери изображения (%)': [lost_percentage_median, lost_percentage_csrt, lost_percentage_mosse],
        'Время работы метода': [total_time_median, total_time_crst, total_time_mosse],
        'Возвращение при выходе за границы экрана (%)': [recovery_rate_median, recovery_rate_crst, recovery_rate_mosse]
    }
    df_results = pd.DataFrame(results)
    pd.set_option('display.max_columns', None)
    print(df_results)