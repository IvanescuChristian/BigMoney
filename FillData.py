import pandas as pd
import os
import sys
from datetime import datetime, timedelta
import numpy as np

BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "historical_hourly")

def calculate_continuous_correlation_0_1(diff1, diff2):
    if pd.isna(diff1) or pd.isna(diff2): return np.nan 
    if diff1 == 0 or diff2 == 0: return 0.5
    cosine_sim = (diff1*diff2)/(abs(diff1)*abs(diff2))
    return (cosine_sim+1)/2

def generate_hourly_timestamps(date_str):
    base_date = datetime.strptime(date_str,"%Y-%m-%d")
    return [(base_date+timedelta(hours=h)).strftime("%Y-%m-%d %H:%M:%S") for h in range(24)]

def socal_v_to_f(sv_value):
    if pd.isna(sv_value): return np.nan
    s_val = str(sv_value)
    if s_val == 'None': return np.nan
    if s_val.endswith('0') and s_val != '0':
        try: return float(s_val.rstrip('0'))
        except ValueError: return np.nan
    try: return float(sv_value)
    except ValueError: return np.nan

def calculate_rsi(prices, period=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss.replace(0, np.nan)
    return (100 - (100 / (1 + rs))).fillna(50)

def process_files():
    all_file_names = sorted([f for f in os.listdir(BASE_DIR) if f.endswith('.csv')])
    all_dates = [f.replace('.csv','') for f in all_file_names]
    all_distinct_coins = set()
    for fn in all_file_names:
        try:
            tmp = pd.read_csv(os.path.join(BASE_DIR,fn), usecols=['coin_id'])
            all_distinct_coins.update(tmp['coin_id'].unique())
        except: continue
    print(f"Found {len(all_distinct_coins)} distinct coins across {len(all_dates)} days.")
    df_by_date = {}
    for fn in all_file_names:
        ds = fn.replace('.csv','')
        try:
            df = pd.read_csv(os.path.join(BASE_DIR,fn))
            df['timestamp_utc'] = pd.to_datetime(df['timestamp_utc'])
            df_by_date[ds] = df
        except:
            df_by_date[ds] = pd.DataFrame(columns=['timestamp_utc','coin_id','price_usd','total_volume_usd'])
    real_days = {coin: set() for coin in all_distinct_coins}
    for ds in all_dates:
        df = df_by_date[ds]
        if not df.empty:
            for c in df['coin_id'].unique():
                real_days[c].add(ds)
    # ===== FORWARD PASS =====
    prev_lh = {}; prev_sum = {}
    for i, ds in enumerate(all_dates):
        cdf = df_by_date[ds].copy()
        present = set(cdf['coin_id'].unique()) if not cdf.empty else set()
        for col in ['hourly_increase','price-soc correlation','price-pop']:
            if col not in cdf.columns: cdf[col] = np.nan
        sv_col = f"social_volume_{ds.replace('-','_')}"
        if sv_col not in cdf.columns: cdf[sv_col] = np.nan
        to_fill = [c for c in all_distinct_coins if c not in present and c in prev_lh]
        if to_fill:
            new_rows = []
            for coin in to_fill:
                pd_d = prev_lh[coin]; hi = pd_d['hi']; hv = pd_d['hv']
                ts = generate_hourly_timestamps(ds); price = pd_d['lp']; vol = hv[0] if hv else 0
                sv_p = prev_sum.get(coin,{}).get('sv',np.nan)
                sv_v = f"{socal_v_to_f(sv_p)}0" if pd.notna(socal_v_to_f(sv_p)) else np.nan
                new_rows.append({'timestamp_utc':pd.to_datetime(ts[0]),'coin_id':coin,'price_usd':price,'total_volume_usd':vol,sv_col:sv_v,'hourly_increase':0,'price-pop':0.5,'price-soc correlation':0.5})
                for h in range(1,24):
                    inc = hi[h-1] if h-1<len(hi) else 0; price = max(0.0, price+inc)
                    v = hv[h] if h<len(hv) else 0; dP = price-new_rows[-1]['price_usd']; dV = v-new_rows[-1]['total_volume_usd']
                    new_rows.append({'timestamp_utc':pd.to_datetime(ts[h]),'coin_id':coin,'price_usd':price,'total_volume_usd':v,sv_col:sv_v,'hourly_increase':dP,'price-pop':calculate_continuous_correlation_0_1(dP,dV),'price-soc correlation':0.5})
            if new_rows:
                ndf = pd.DataFrame(new_rows); cols = list(cdf.columns) if not cdf.empty else list(ndf.columns)
                for c in cols:
                    if c not in ndf.columns: ndf[c]=np.nan
                cdf = pd.concat([cdf,ndf[cols]],ignore_index=True).sort_values(['coin_id','timestamp_utc']).reset_index(drop=True)
                df_by_date[ds] = cdf
        cd_med = cdf.groupby('coin_id')['price_usd'].median() if not cdf.empty else pd.Series()
        cd_sv = cdf.groupby('coin_id')[sv_col].first() if sv_col in cdf.columns and not cdf.empty else pd.Series()
        parts = []
        for coin, grp in cdf.groupby('coin_id'):
            grp['hourly_increase'] = grp['price_usd'].diff(); grp['price-pop'] = np.nan
            if not grp.empty:
                fi = grp.index[0]
                if coin in prev_lh:
                    lp=prev_lh[coin]['lp']; lv=prev_lh[coin]['lv']
                    grp.loc[fi,'hourly_increase']=grp.iloc[0]['price_usd']-lp; grp.loc[fi,'price-pop']=calculate_continuous_correlation_0_1(grp.iloc[0]['price_usd']-lp,grp.iloc[0]['total_volume_usd']-lv)
                else: grp.loc[fi,'hourly_increase']=0; grp.loc[fi,'price-pop']=0.5
                for j in range(1,len(grp)):
                    ci,pi=grp.index[j],grp.index[j-1]; grp.loc[ci,'price-pop']=calculate_continuous_correlation_0_1(grp.loc[ci,'price_usd']-grp.loc[pi,'price_usd'],grp.loc[ci,'total_volume_usd']-grp.loc[pi,'total_volume_usd'])
            grp['price-soc correlation']=0.5
            if sv_col in grp.columns and coin in prev_sum:
                cm=cd_med.get(coin);cs=cd_sv.get(coin);pm=prev_sum[coin]['mp'];ps=prev_sum[coin]['sv']
                if pd.notna(cm) and pd.notna(cs) and pd.notna(pm) and pd.notna(ps):
                    svc=socal_v_to_f(cs);svp=socal_v_to_f(ps)
                    if pd.notna(svc) and pd.notna(svp): grp.loc[:,'price-soc correlation']=calculate_continuous_correlation_0_1(cm-pm,svc-svp)
            parts.append(grp)
        if parts: cdf = pd.concat(parts).sort_values(['coin_id','timestamp_utc']).reset_index(drop=True); df_by_date[ds] = cdf
        prev_lh.clear(); prev_sum.clear()
        for coin, grp in cdf.groupby('coin_id'):
            if not grp.empty:
                lr=grp.iloc[-1]; hi=grp['hourly_increase'].tolist(); hv=grp['total_volume_usd'].tolist()
                while len(hi)<24: hi.append(0)
                while len(hv)<24: hv.append(0)
                prev_lh[coin]={'lp':lr['price_usd'],'lv':lr['total_volume_usd'],'hi':hi,'hv':hv}
            prev_sum[coin]={'mp':cd_med.get(coin),'sv':cd_sv.get(coin)}
    # ===== BACKWARD PASS =====
    for i in range(len(all_dates)-2,-1,-1):
        ds = all_dates[i]; ns = all_dates[i+1]
        print(f"Backward Pass: {ds} <- {ns}")
        cdf = df_by_date[ds].copy(); ndf = df_by_date[ns]
        present = set(cdf['coin_id'].unique()) if not cdf.empty else set()
        np_ = set(ndf['coin_id'].unique()) if not ndf.empty else set()
        to_fill = [c for c in all_distinct_coins if c not in present and c in np_]
        if to_fill:
            sv_col = f"social_volume_{ds.replace('-','_')}"; new_rows = []
            for coin in to_fill:
                nc = ndf[ndf['coin_id']==coin].sort_values('timestamp_utc')
                if nc.empty: continue
                nhi = nc['hourly_increase'].tolist(); nhv = nc['total_volume_usd'].tolist()
                while len(nhi)<24: nhi.append(0.0)
                while len(nhv)<24: nhv.append(0.0)
                nsvc = f"social_volume_{ns.replace('-','_')}"; nsv = nc[nsvc].iloc[0] if nsvc in nc.columns else np.nan
                ts = generate_hourly_timestamps(ds); prices = [0.0]*24
                prices[23] = max(0.0, nc.iloc[0]['price_usd'] - nhi[0])
                for h in range(22,-1,-1): prices[h] = max(0.0, prices[h+1] - (nhi[h+1] if h+1<len(nhi) else 0.0))
                for h,t in enumerate(ts):
                    hiv = prices[h]-prices[h-1] if h>0 else 0.0; svf = socal_v_to_f(nsv)
                    new_rows.append({'timestamp_utc':pd.to_datetime(t),'coin_id':coin,'price_usd':prices[h],'total_volume_usd':nhv[h] if h<len(nhv) else 0.0,sv_col:f"{svf}0" if pd.notna(svf) else np.nan,'hourly_increase':hiv,'price-pop':np.nan,'price-soc correlation':np.nan})
            if new_rows:
                ndf2 = pd.DataFrame(new_rows); cols = list(cdf.columns) if not cdf.empty else list(ndf2.columns)
                for c in cols:
                    if c not in ndf2.columns: ndf2[c]=np.nan
                cdf = pd.concat([cdf,ndf2[cols]],ignore_index=True).sort_values(['coin_id','timestamp_utc']).reset_index(drop=True); df_by_date[ds] = cdf
    # ===== INTERPOLATION PASS: fix continuity =====
    # Daca forward fill a pus preturi in zilele X dar apoi vine o zi reala Y,
    # preturile fill-uite pot sa NU se lege de Y. Le rescriem ca interpolare
    # liniara intre ultimul pret REAL si primul pret REAL urmator.
    print("\nInterpolation pass: fixing continuity gaps...")
    fixed_count = 0
    for coin in all_distinct_coins:
        real = sorted(real_days[coin])
        if len(real) < 2: continue
        for r_idx in range(len(real)-1):
            sd = real[r_idx]; ed = real[r_idx+1]
            si = all_dates.index(sd); ei = all_dates.index(ed)
            gap = ei - si - 1
            if gap <= 0: continue
            sdf = df_by_date[sd]; s_c = sdf[sdf['coin_id']==coin].sort_values('timestamp_utc')
            edf = df_by_date[ed]; e_c = edf[edf['coin_id']==coin].sort_values('timestamp_utc')
            if s_c.empty or e_c.empty: continue
            p_start = s_c.iloc[-1]['price_usd']; p_end = e_c.iloc[0]['price_usd']
            total_h = gap * 24
            for gi in range(gap):
                gd = all_dates[si+1+gi]; gdf = df_by_date[gd]; mask = gdf['coin_id']==coin
                if mask.sum()==0: continue
                gc = gdf.loc[mask].sort_values('timestamp_utc')
                for hi in range(len(gc)):
                    hp = gi*24+hi+1; t = hp/(total_h+1); new_p = max(0.0, p_start+(p_end-p_start)*t)
                    gdf.loc[gc.index[hi],'price_usd'] = new_p
                df_by_date[gd] = gdf; fixed_count += 1
    if fixed_count: print(f"  Fixed {fixed_count} coin-day gaps")
    # ===== FINAL PASS: recalculate + RSI + volatility + save =====
    print("\nFinal pass: recalculate + save...")
    prev_lh = {}; prev_sum = {}
    for ds in all_dates:
        fpath = os.path.join(BASE_DIR, f"{ds}.csv")
        cdf = df_by_date[ds].copy(); sv_col = f"social_volume_{ds.replace('-','_')}"
        if cdf.empty:
            pd.DataFrame(columns=['timestamp_utc','coin_id','price_usd','total_volume_usd',sv_col,'hourly_increase','price-soc correlation','price-pop','RSI','volatility']).to_csv(fpath, index=False); continue
        if sv_col not in cdf.columns: cdf[sv_col] = np.nan
        cd_med = cdf.groupby('coin_id')['price_usd'].median()
        cd_sv = cdf.groupby('coin_id')[sv_col].first()
        parts = []
        for coin, grp in cdf.groupby('coin_id'):
            grp = grp.sort_values('timestamp_utc').copy()
            grp['hourly_increase'] = grp['price_usd'].diff(); grp['price-pop'] = np.nan
            if not grp.empty:
                fi = grp.index[0]
                if coin in prev_lh:
                    lp=prev_lh[coin]['lp']; lv=prev_lh[coin]['lv']; grp.loc[fi,'hourly_increase']=grp.iloc[0]['price_usd']-lp; grp.loc[fi,'price-pop']=calculate_continuous_correlation_0_1(grp.iloc[0]['price_usd']-lp,grp.iloc[0]['total_volume_usd']-lv)
                else: grp.loc[fi,'hourly_increase']=0; grp.loc[fi,'price-pop']=0.5
                for j in range(1,len(grp)):
                    ci,pi=grp.index[j],grp.index[j-1]; grp.loc[ci,'price-pop']=calculate_continuous_correlation_0_1(grp.loc[ci,'price_usd']-grp.loc[pi,'price_usd'],grp.loc[ci,'total_volume_usd']-grp.loc[pi,'total_volume_usd'])
            grp['price-soc correlation']=0.5
            if sv_col in grp.columns and coin in prev_sum:
                cm=cd_med.get(coin);cs=cd_sv.get(coin);pm=prev_sum[coin]['mp'];ps=prev_sum[coin]['sv']
                if pd.notna(cm) and pd.notna(cs) and pd.notna(pm) and pd.notna(ps):
                    svc=socal_v_to_f(cs);svp=socal_v_to_f(ps)
                    if pd.notna(svc) and pd.notna(svp): grp.loc[:,'price-soc correlation']=calculate_continuous_correlation_0_1(cm-pm,svc-svp)
            grp['RSI'] = calculate_rsi(grp['price_usd'])
            pp = grp['price_usd'].pct_change().replace([np.inf,-np.inf],0).fillna(0)
            grp['volatility'] = pp.rolling(24,min_periods=1).std().fillna(0)
            parts.append(grp)
        if parts:
            final = pd.concat(parts).sort_values(['coin_id','timestamp_utc']).reset_index(drop=True)
            prev_lh.clear(); prev_sum.clear()
            for coin, grp in final.groupby('coin_id'):
                if not grp.empty:
                    lr=grp.iloc[-1]; hi=grp['hourly_increase'].tolist(); hv=grp['total_volume_usd'].tolist()
                    while len(hi)<24: hi.append(0)
                    while len(hv)<24: hv.append(0)
                    prev_lh[coin]={'lp':lr['price_usd'],'lv':lr['total_volume_usd'],'hi':hi,'hv':hv}
                prev_sum[coin]={'mp':cd_med.get(coin),'sv':cd_sv.get(coin)}
            save_cols = ['timestamp_utc','coin_id','price_usd','total_volume_usd','hourly_increase','price-soc correlation','price-pop',sv_col,'RSI','volatility']
            for c in save_cols:
                if c not in final.columns: final[c]=0.0
            final[save_cols].to_csv(fpath, index=False)
            print(f"  Saved {ds} ({final['coin_id'].nunique()} coins)")
    print("\nDone.")

if __name__ == "__main__":
    process_files()
