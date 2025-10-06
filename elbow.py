import matplotlib.pyplot as plt
import pandas as pd

def plot_elbow():

    start_date = 0

    df = pd.read_csv("elbow_data.txt", delimiter=',', names=('ext','flex'))
    df["range"] = df["flex"] - df["ext"]

    days = [i + start_date for i in range(len(df))]

    fig, ax = plt.subplots(figsize=(12,9))

    plt.ylim(0,150)
    plt.xlim(0,len(df)+start_date)
    plt.xlabel("Time [days]")
    plt.ylabel("Range [degrees]")

    ax.set_yticks(range(0, 151, 10))
    ax.grid(axis="y", linestyle="-", alpha=0.7)

    plt.plot(days, df['ext'], color='red')
    plt.plot(days, df['flex'], color='red')
    plt.plot(days, df['range'], color='blue')

    ax.fill_between(days, df['ext'], df['flex'], color="red", alpha=0.3)

    plt.savefig('plot.png')

if __name__ == "__main__":
    plot_elbow()
