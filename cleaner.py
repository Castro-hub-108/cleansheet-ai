import pandas as pd

def clean_data(file):
    df = pd.read_csv(file)
    original_shape = df.shape

    df.drop_duplicates(inplace=True)
    df.fillna(method='ffill', inplace=True)

    cleaned_shape = df.shape
    log = {
        "Original rows": original_shape[0],
        "Cleaned rows": cleaned_shape[0],
        "Duplicates removed": original_shape[0] - cleaned_shape[0],
        "Missing values filled": df.isnull().sum().sum()
    }
    return df, log
