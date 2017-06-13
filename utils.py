import pandas as pd
from pandas import ExcelWriter
import numpy as np
import math
from tqdm import tqdm
import math as m

one_fun = lambda x: 1.0
one_fun_nan = lambda x: np.nan if np.isnan(x) else 1.0
nan_fun = lambda x: np.nan


def get_data_on_dates(df, dates):
    dts = df.index
    nr = len(dts)
    idx = dts.searchsorted(dates, side='left')
    idx = [x if x < nr else nr - 1 for x in idx]
    df_ret = df.ix[idx, :]
    df_ret.index = dates
    null_rows = dates < dts[0]
    df_ret.loc[dates[null_rows], :] = np.nan
    return df_ret


def sigmoid(x, val=1):
    return 2.0/(1.0+math.exp(-val*x))


def portfolio_allocation(signals, prices, weighing_scheme="LO_Sigmoid",
                         vol_trg=False, vol_period=66, sigmoid_val=1, norm_row=False):
    dates = prices.index
    signals_ = get_data_on_dates(signals, dates)

    if vol_trg:
        ret = prices.pct_change()
        vols = ret.rolling(center=False, window=vol_period).std()
        vols_ = get_data_on_dates(vols, dates)
    else:
        vols_ = prices.applymap(one_fun)

    bench_allocation = signals_.applymap(one_fun_nan)
    #bench_allocation = bench_allocation/vols_

    bench_allocation = bench_allocation.apply(lambda x: x/np.sum(x), axis=1)
    bench_allocation = bench_allocation.fillna(0)
    trd_sig = signals_.copy()
    if weighing_scheme == "LO_Sigmoid":
        trd_sig = signals_.applymap(lambda x: sigmoid(x, sigmoid_val))
    elif weighing_scheme == "LS_Sigmoid":
        trd_sig = signals_.applymap(lambda x: sigmoid(x, sigmoid_val) - 1.0)
    elif weighing_scheme == "LS_Sign":
        trd_sig = signals_.applymap(lambda x: np.sign(x))
    elif weighing_scheme == "LS":
        trd_sig = signals_

    allocation = trd_sig * bench_allocation/ vols_
    if norm_row:
        allocation = allocation.apply(lambda x: x / np.sum(x), axis=1)
    allocation = allocation.fillna(0)

    return [allocation, bench_allocation]


def performance_analysis(prices, allocation, cost=0.0, freq=1):
    n_dates = allocation.shape[0]
    alloc_ = allocation.copy()
    dates = allocation.index
    prices_ = get_data_on_dates(prices, dates)
    rets = prices_.pct_change()
    pnl = rets.applymap(nan_fun)

    for i in tqdm(range(1, n_dates)):
        pnl.ix[i, :] = alloc_.ix[i-1, :].values * rets.ix[i, :].values - abs(cost)*np.abs(alloc_.ix[i-1, :].values - alloc_.ix[i, :].values)

    pnl["Combined"] = np.sum(pnl, 1) #row wise sum
    cummPnl = np.cumsum(pnl, axis=0)
    means = np.mean(pnl, axis=0) * (250.0/freq)
    stdev = np.std(pnl, axis=0) * math.sqrt(250/freq)
    sharpe = means/stdev
    return {"allocation": alloc_, "pnl": pnl, "cummpnl": cummPnl, "means": means, "sdev": stdev, "Sharpe": sharpe}


def get_sharpe(rets, signals, cost=0.0, freq=1):
    sig = signals.applymap(np.sign)
    sig_ = sig.shift(1).fillna(0)
    pnl = rets * sig_ - abs(cost)*np.abs(sig_ - sig)
    pnl = pnl.fillna(0)
    means = np.mean(pnl, axis=0) * (250.0 / freq)
    stdev = np.std(pnl, axis=0) * math.sqrt(250 / freq)
    sharpe = means / stdev
    return sharpe


def performance_comparision(strategy, benchmark):
    perf_comparision = pd.DataFrame(data=strategy["cummpnl"]["Combined"])
    perf_comparision["Benchmark"] = benchmark["cummpnl"]["Combined"]

    results_df = pd.DataFrame()
    results_df["Strat mean"] = strategy["means"]
    results_df["Strat sdev"] = strategy["sdev"]
    results_df["Strat Sharpe"] = strategy["Sharpe"]
    results_df["Bench mean"] = benchmark["means"]
    results_df["Bench sdev"] = benchmark["sdev"]
    results_df["Bench Sharpe"] = benchmark["Sharpe"]
    results_df["Diff Sharpe"] = strategy["Sharpe"] - benchmark["Sharpe"]

    return {"cumm_df": perf_comparision, "results_df": results_df}


def save_results(perf_comparision, xls_filename, sheet_str=""):
    writer = ExcelWriter(xls_filename)
    sheet_name = sheet_str
    perf_comparision["results_df"].to_excel(writer, "R_" + sheet_name)
    perf_comparision["cumm_df"].to_excel(writer, "C_" + sheet_name)
    writer.save()


def rolling_backtest_index(nsamples, look_back, roll_ahead):
    coll = []
    for i in range(look_back + roll_ahead, nsamples + roll_ahead, roll_ahead):
        col = [i - look_back - roll_ahead, i - roll_ahead, min(i, nsamples)]
        coll.append(col)

    return coll


def vol_adj(cumm_df, ann_vol=0.1):
    pnl_df = cumm_df.diff(1)
    pnl_df = ann_vol * pnl_df/(m.sqrt(250)*pnl_df.std())
    return pnl_df.cumsum()
