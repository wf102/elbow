import numpy as np
import pandas as pd
from filterpy.kalman import KalmanFilter, rts_smoother
from filterpy.common import Q_discrete_white_noise

def kalman_smooth_rom(series, measurement_sd=1.0, process_sd=0.1):
    """
    Smooth a scalar time series with a constant-velocity Kalman filter + RTS smoother.

    One sample per day, gaps allowed as NaN.

    Missing days are handled as predict-only steps (no update), so the
    smoother still produces an estimate for them without needing to
    interpolate the raw data first.
    """
    zs = [None if pd.isna(v) else float(v) for v in series]

    first_val = next((z for z in zs if z is not None), 0.0)

    kf = KalmanFilter(dim_x=2, dim_z=1)
    kf.x = np.array([[first_val], [0.0]])
    kf.F = np.array([[1.0, 1.0],
                      [0.0, 1.0]])
    kf.H = np.array([[1.0, 0.0]])
    kf.P *= 100.0
    kf.R = measurement_sd ** 2
    kf.Q = Q_discrete_white_noise(dim=2, dt=1.0, var=process_sd ** 2)

    means, covariances, _, _ = kf.batch_filter(zs)
    Fs = [kf.F] * len(zs)
    Qs = [kf.Q] * len(zs)
    smoothed_means, _, _, _ = rts_smoother(means, covariances, Fs, Qs)

    return pd.Series(smoothed_means[:, 0, 0], index=series.index)