import pandas


if __name__ == "__main__": 
    df = pandas.read_csv("data.csv")
    train_df = df.sample(frac=0.8, random_state=42)
    val_df = df.drop(train_df.index)
    train_df.to_csv("train_dataset.csv", index=False)
    val_df.to_csv("val_df.csv", index=False)


