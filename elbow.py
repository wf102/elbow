import matplotlib.pyplot as plt
import pandas as pd

WINDOW = 5
min_periods = 2

def plot_elbow(window=1):

    start_date = 0

    df = pd.read_csv("elbow_data.txt", delimiter=',', names=('ext','flex'))

    df["mid"] = 0.5 * (df["flex"] + df["ext"])
    df["range"] = df["flex"] - df["ext"]

    df["flex_ma"] = df["flex"].rolling(window=window, center=True, min_periods=min_periods).mean()
    df["ext_ma"] = df["ext"].rolling(window=window, center=True, min_periods=min_periods).mean()

    df["mid_ma"] = df["mid"].rolling(window=window, center=True, min_periods=min_periods).mean()
    df["range_ma"] = df["range"].rolling(window=window, center=True, min_periods=min_periods).mean()

    days = [i + start_date for i in range(len(df))]

    fig, ax = plt.subplots(figsize=(12,9))

    plt.ylim(0,150)
    plt.xlim(0,len(df)+start_date)
    plt.xlabel("Time [days]")
    plt.ylabel("Range [degrees]")

    ax.set_yticks(range(0, 151, 10))
    ax.grid(axis="y", linestyle="-", alpha=0.7)

    plt.scatter(days, df['ext'], color='red', marker='.', s=6)
    plt.scatter(days, df['flex'], color='red', marker='.', s=6)
    plt.plot(days, df['range_ma'], color='blue')

    ax.fill_between(days, df['ext_ma'], df['flex_ma'], color="red", alpha=0.3)

    plt.savefig('plot.png')

if __name__ == "__main__":
    plot_elbow(WINDOW)