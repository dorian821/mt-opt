

dets = web('SPY',yahoo,start,end)

raw_data = os.listdir(data_dir)
error_df = pd.DataFrame()
for name in raw_data:
  df = pd.read_csv(data_dir+name,'rb',sep=',')
  err_df = df[(df['trade_volume']>0)&(df['open']==0)]
  error_df = pd.concat([error_df,err_df],axis=0)

logs = os.listdir(log_dir)
awol_report = pd.DataFrame()
for name in logs:
  df = pd.read_csv(log_dir+name,'rb',sep=',',names={'Files'})
  dats = pd.Series([pd.to_datetime(x.split('_',5)[5],infer_datetime_format=True) for x in df['Files']])
  awol = list(set(dets.index)
 
