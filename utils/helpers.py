import pandas as pd


def generate_imputed_date_datasets():
    for task in ['train', 'test']:
        df = pd.read_csv(f'data/{task}.csv')
        calendar = pd.read_csv(f'data/{task}_calendar.csv')
        df.drop(columns=[i for i in df.columns if i not in ['orders', 'date', 'warehouse']], inplace=True)
        df = pd.merge(calendar, df, on=['date', 'warehouse'], how='left')
        df['holiday_name'] = df['holiday_name'].fillna('none')
        try:
            df['orders'] = df['orders'].fillna(-1)
        except KeyError:
            pass
        df.to_csv(f'data/{task}_imputed_dates.csv', index=False)
