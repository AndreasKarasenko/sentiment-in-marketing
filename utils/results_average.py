import pandas as pd

data_f1 = "./results/overview/summary_all_F1.csv"
df_f1 = pd.read_csv(data_f1)

data_rec = "./results/overview/summary_all_REC.csv"

data_prec = "./results/overview/summary_all_PREC.csv"

data_acc = "./results/overview/summary_all_ACC.csv"


def dropnas(df):
    df = df.dropna()
    return df