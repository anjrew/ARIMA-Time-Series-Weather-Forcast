import pandas as pd

def format_data_frame(df: pd.DataFrame, set_index_time_step=True, first_date=None, drop_cols=[]) -> pd.DataFrame:
    df = df.set_axis(list(map(lambda cl: cl.strip() ,df.columns)), axis=1)
    df.index.name = 'DATE'
    
    assert len(pd.date_range(df.index.min(), df.index.max()).difference(df.index)) == 0, "Missing values in date range"
    
    if first_date is not None:
        df = df[df.index.year > first_date]  # type: ignore
    
    df = df.rename({'TG': 'temp_c'}, axis=1)
    # Change tenths to full integer degrees
    df['temp_c'] = df['temp_c'] / 10
    df['month'] = df.index.month  # type: ignore
    if set_index_time_step is True:
        df['time_step'] = range(len(df))
        df = df.reset_index()
        df = df.set_index('time_step')
    
    df = df.drop(drop_cols, axis=1)
    return df