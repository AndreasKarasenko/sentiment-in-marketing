from utils.dataloader import googleplay
from utils.describe import describe

df = googleplay(config=None, path="./data/ikea.csv")
print(describe(df, text_col="content", score_col="score"))