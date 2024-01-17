# a short script that loads the amazon review dataset and prints some descriptive statistics
from utils.dataloader import amazon
from utils.describe import describe

df = amazon(config=None, path="./data/Luxury_Beauty_5.json.gz")
print(describe(df, text_col="reviewText", score_col="overall"))