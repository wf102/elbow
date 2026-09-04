import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import datetime
from regularisation import kalman_smooth_rom

FILENAME = "elbow_data.txt"
measurement_sd = 1.0

accident_date = datetime.datetime(2025, 8, 9)
orif_date = datetime.datetime(2025, 8, 29)
first_recorded_date = datetime.datetime(2025, 9, 12)
arthrolysis_date = datetime.datetime(2026, 3, 26)

def read_data(filename):
    df = pd.read_csv(filename, delimiter=',', names=('ext','flex'))
    df["mid"] = 0.5 * (df["flex"] + df["ext"])
    df["range"] = df["flex"] - df["ext"]
    df["flex_smooth"] = kalman_smooth_rom(df["flex"], measurement_sd)
    df["ext_smooth"] = kalman_smooth_rom(df["ext"], measurement_sd)
    df["range_smooth"] = df["flex_smooth"] - df["ext_smooth"]

    df["date"] = [first_recorded_date + datetime.timedelta(days=i) for i in range(len(df))]

    return df

def _mdate2days(x):
    x_arr = np.asarray(x)
    if x_arr.ndim == 0:
        dt = mdates.num2date(float(x_arr)).replace(tzinfo=None)
        return (dt - orif_date).days
    values = [
        (mdates.num2date(float(xi)).replace(tzinfo=None) - orif_date).days
        for xi in x_arr.ravel()
    ]
    return np.array(values).reshape(x_arr.shape)

def _days2mdate(x):
    x_arr = np.asarray(x)
    if x_arr.ndim == 0:
        return mdates.date2num(orif_date + datetime.timedelta(days=int(x_arr)))
    values = [
        mdates.date2num(orif_date + datetime.timedelta(days=int(xi)))
        for xi in x_arr.ravel()
    ]
    return np.array(values).reshape(x_arr.shape)

def plot_elbow(df):

    last_date = df["date"].iloc[-1].to_pydatetime() + datetime.timedelta(days=14)

    fig, ax = plt.subplots(figsize=(14,8))

    plt.ylim(0,150)
    plt.xlim(accident_date, last_date)
    ax.set_xlabel("Date")
    ax.set_ylabel("Range [degrees]")
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

    max_days = (last_date - orif_date).days
    secax = ax.secondary_xaxis('top', functions=(_mdate2days, _days2mdate))
    secax.set_xticks(range(0, max_days + 1, 20))

    ax.set_yticks(range(0, 151, 10))
    ax.grid(axis="y", linestyle="-", alpha=0.7)

    plt.scatter(df["date"], df['ext'], color='red', marker='.', s=6)
    plt.scatter(df["date"], df['flex'], color='red', marker='.', s=6)
    plt.plot(df["date"], df['range_smooth'], color='blue')

    plt.axvline(x = orif_date, color = 'darkgrey', linestyle='--')
    plt.axvline(x = arthrolysis_date, color = 'darkgrey', linestyle='--')
    ax.text(orif_date+datetime.timedelta(days=-4), 12, 'ORIF', color='darkgrey', rotation=90)
    ax.text(arthrolysis_date+datetime.timedelta(days=-4), 12, 'Arthrolysis', color='darkgrey', rotation=90)

    ax.fill_between(df["date"], df['ext_smooth'], df['flex_smooth'], color="red", alpha=0.3)
    ax.tick_params(axis='x', rotation=0)

    plt.tight_layout()
    plt.savefig('plot.png')

if __name__ == "__main__":
    df = read_data(FILENAME)
    plot_elbow(df)
