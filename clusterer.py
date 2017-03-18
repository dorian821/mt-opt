
1#oscilltor matrix
   # normalize data to look at distance from recent extreme and  divergence
#sma matrix
#ew matrix

#subtract row from matrix
#sum all diff rows
#take closest and all within a margin from the closest, eg 1.5ppp
we can adjust it to identify direction only to suit our 10d needs





def clusterer(data, p, m):
    diff_mx = np.abs(data.iloc[-1] - data.ix[:-1])
    day_s = diff_mx.sum(axis=1).sort()
    eps = day_s.min() * (1+p)
    eps_slope = days_s.rolling(window=3,center=True).apply(three_linest)
    cluster = data.index[(day_s<eps) & (eps_slope<m)]
    return cluster

clusters = [osc_cluster = clusterer(osc_df), sma_cluster = clusterer(sma_df), ew_cluster = clusterer(ew_df)]
for c in clusters:
    if len(c) >= 20:
        dir = stk['10d_dir'][list(c)]
        buy = stk['D2Op[]
        sell = stk[stk.index == c]
        





