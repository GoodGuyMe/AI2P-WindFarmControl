import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

agents = ["A2C_1", "PPO_1", "TRPO_1"]

def clean_data(df):
    for x in df.index:
        if df.loc[x, "Step"] > 200000:
            df.drop(x, inplace=True)
    return df

def gather_data(path):
    df = pd.read_csv(path)
    return clean_data(df)

def plot(dir):
    dfs = [(gather_data(f"{dir}/{agent}.csv"), agent.split("_")[0]) for agent in agents]

    for df, agent in dfs:
        x = df["Step"]
        y = df["Value"]
        plt.plot(x, y, label=agent)
    x = [0, 200000]
    y = [0.78, 0.78]
    plt.plot(x, y, label="Greedy")
    x = [0, 200000]
    y = [0.7136, 0.7136]
    plt.plot(x, y, label="Floris")
    plt.title(dir)
    plt.legend()
    plt.xlabel("Step")
    plt.ylabel("Value")
    plt.show()

def stats(dir):
    dfs = [(gather_data(f"{dir}/{agent}.csv"), agent.split("_")[0]) for agent in agents]

    for df, agent in dfs:
        mean = df.nlargest(50, "Step")["Value"].mean()
        max = df["Value"].max()
        print(f"Max value for {agent}: {max}")
        print(f"Mean value of last 100 steps for {agent}: {mean}")

if __name__ == '__main__':
    # plot("Predicted Power")
    plot("Validation Power")

    # stats("Predicted Power")
    stats("Validation Power")