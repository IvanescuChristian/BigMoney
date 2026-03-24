import os
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta
import glob
import warnings

warnings.filterwarnings('ignore')

INPUT_DIR = os.path.join(os.getcwd(), 'historical_hourly')
OUTPUT_DIR = os.path.join(os.getcwd(), 'predicted_hourly')
os.makedirs(OUTPUT_DIR, exist_ok=True)

HORIZON = 24  # 24 ore = 1 zi de predictie
NUM_SIMULATIONS = 50

FEATURES = ['total_volume_usd', 'social_volume', 'hourly_increase', 'price-pop',
            'price-soc correlation', 'RSI', 'volatility']

# ========== INCARCARE DATE ==========
def load_all_data():
    all_files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.csv")))
    if not all_files: return None
    df_list = []
    for f in all_files:
        try:
            tmp = pd.read_csv(f)
            sv_col = [c for c in tmp.columns if 'social_volume' in c]
            tmp['social_volume'] = tmp[sv_col[0]].fillna(0.0) if sv_col else 0.0
            needed = ['timestamp_utc','coin_id','price_usd','total_volume_usd',
                      'social_volume','hourly_increase','price-soc correlation',
                      'price-pop','RSI','volatility']
            for col in needed:
                if col not in tmp.columns: tmp[col] = 0.0
            df_list.append(tmp[needed])
        except: pass
    if not df_list: return None
    full = pd.concat(df_list, ignore_index=True)
    full['timestamp_utc'] = pd.to_datetime(full['timestamp_utc'])
    full = full.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    print(f"    -> {len(all_files)} fisiere, {len(full)} randuri totale.")
    return full.sort_values(['coin_id','timestamp_utc'])

# ========== RSI / VOLATILITATE DINAMICE pt Monte Carlo ==========
def dynamic_rsi(price_mat, period=14):
    if price_mat.shape[1] < period + 1: return np.full(price_mat.shape[0], 50.0)
    deltas = np.diff(price_mat[:, -period-1:], axis=1)
    ups = np.sum(np.where(deltas > 0, deltas, 0), axis=1) / period
    downs = np.sum(np.where(deltas < 0, -deltas, 0), axis=1) / period
    downs = np.where(downs == 0, 1e-9, downs)
    rsi = 100.0 - (100.0 / (1.0 + (ups / downs)))
    return np.nan_to_num(rsi, nan=50.0, posinf=100.0, neginf=0.0)

def dynamic_volat(price_mat, period=24):
    if price_mat.shape[1] < period: return np.zeros(price_mat.shape[0])
    arr = price_mat[:, -period:]
    with np.errstate(divide='ignore', invalid='ignore'):
        rets = np.diff(arr, axis=1) / (arr[:, :-1] + 1e-9)
    return np.nan_to_num(np.std(rets, axis=1), nan=0.0)

# ========== RULARE ==========
def run_prediction():
    print(">>> [1] Incarcare date istorice...")
    df = load_all_data()
    if df is None or df.empty: return

    coins = df['coin_id'].unique()
    all_results = []
    summary_results = []
    error_matrix_rows = []

    print(f">>> [2] Predictie duala (XGBoost pret + MLP directie) pentru {len(coins)} monede...\n")

    for coin in coins:
        c_df = df[df['coin_id'] == coin].copy().sort_values('timestamp_utc')

        # Return procentual ca target XGBoost
        c_df['pct_return'] = c_df['price_usd'].pct_change().replace([np.inf,-np.inf],0).fillna(0)
        # Label binar: 1=crestere, 0=scadere (target MLP perceptron)
        c_df['direction'] = (c_df['pct_return'] > 0).astype(int)
        # Momentum features
        c_df['momentum_6h'] = c_df['price_usd'].pct_change(6).replace([np.inf,-np.inf],0).fillna(0)
        c_df['momentum_24h'] = c_df['price_usd'].pct_change(24).replace([np.inf,-np.inf],0).fillna(0)
        c_df['vol_change'] = c_df['total_volume_usd'].pct_change().replace([np.inf,-np.inf],0).fillna(0)

        train = c_df.replace([np.inf,-np.inf],0).fillna(0)
        if len(train) < 50:
            print(f"    ! Skip {coin} (prea putine date)")
            continue

        X_cols = FEATURES + ['momentum_6h','momentum_24h','vol_change']
        X = train[X_cols]
        y_price = train['pct_return']
        y_dir = train['direction']

        # --- Model 1: XGBoost pentru return procentual ---
        model_xgb = xgb.XGBRegressor(
            n_estimators=400, learning_rate=0.04, max_depth=6,
            subsample=0.8, colsample_bytree=0.8, n_jobs=-1
        )
        try:
            model_xgb.fit(X, y_price)
        except Exception as e:
            print(f"    ! XGBoost fail {coin}: {e}")
            continue

        # --- Model 2: MLP Perceptron pentru directie (crestere/scadere) ---
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        model_mlp = MLPClassifier(
            hidden_layer_sizes=(64, 32), activation='relu',
            max_iter=500, random_state=42, early_stopping=True
        )
        try:
            model_mlp.fit(X_scaled, y_dir)
            mlp_train_acc = model_mlp.score(X_scaled, y_dir)
        except Exception as e:
            print(f"    ! MLP fail {coin}: {e}")
            model_mlp = None
            mlp_train_acc = 0.0

        print(f"    -> {coin}: XGBoost OK, MLP acc={mlp_train_acc:.3f}")

        # --- Monte Carlo forward simulation ---
        last_row = c_df.iloc[-1]
        t_curr = last_row['timestamp_utc']

        base_hist = np.array(c_df['price_usd'].tail(50).tolist())
        price_mat = np.tile(base_hist, (NUM_SIMULATIONS, 1))

        s_vol = last_row['total_volume_usd']
        s_soc = last_row['social_volume']
        s_pop = last_row['price-pop']
        s_corr = last_row['price-soc correlation']

        hist_rets = train['pct_return'].values
        hist_rets = hist_rets[hist_rets != 0]
        hist_std = np.std(hist_rets) if len(hist_rets) >= 10 else 0.001

        coin_prices = []
        coin_directions = []

        for h in range(HORIZON):
            t_curr += timedelta(hours=1)
            l1 = price_mat[:, -1]
            inc = l1 - price_mat[:, -2]

            rsi_vec = dynamic_rsi(price_mat)
            vol_vec = np.maximum(dynamic_volat(price_mat), 0.0015)

            mom6 = (l1/(price_mat[:,-6]+1e-9))-1 if price_mat.shape[1]>=6 else np.zeros(NUM_SIMULATIONS)
            mom24 = (l1/(price_mat[:,-24]+1e-9))-1 if price_mat.shape[1]>=24 else np.zeros(NUM_SIMULATIONS)

            inp_df = pd.DataFrame({
                'total_volume_usd': np.full(NUM_SIMULATIONS, s_vol),
                'social_volume': np.full(NUM_SIMULATIONS, s_soc),
                'hourly_increase': inc,
                'price-pop': np.full(NUM_SIMULATIONS, s_pop),
                'price-soc correlation': np.full(NUM_SIMULATIONS, s_corr),
                'RSI': rsi_vec, 'volatility': vol_vec,
                'momentum_6h': mom6, 'momentum_24h': mom24,
                'vol_change': np.zeros(NUM_SIMULATIONS)
            })[X_cols]

            # XGBoost: predicted return
            pred_returns = model_xgb.predict(inp_df)
            noise = np.random.normal(0, hist_std * 0.5, NUM_SIMULATIONS)
            final_returns = pred_returns + noise
            final_prices = l1 * (1 + final_returns)
            final_prices = np.maximum(final_prices, 1e-9)

            mean_price = np.mean(final_prices)
            mean_inc = mean_price - np.mean(l1)

            # MLP: predicted direction
            if model_mlp is not None:
                mean_features = inp_df.mean().values.reshape(1, -1)
                mean_scaled = scaler.transform(mean_features)
                dir_pred = model_mlp.predict(mean_scaled)[0]
                dir_proba = model_mlp.predict_proba(mean_scaled)[0]
                confidence = max(dir_proba)
            else:
                dir_pred = 1 if mean_inc > 0 else 0
                confidence = 0.5

            price_mat = np.column_stack((price_mat[:, 1:], final_prices))
            coin_prices.append(mean_price)
            coin_directions.append(dir_pred)

            all_results.append({
                'timestamp_utc': t_curr,
                'coin_id': coin,
                'price_usd': mean_price,
                'total_volume_usd': s_vol,
                'hourly_increase': mean_inc,
                'price-pop': s_pop
            })

        # Sumar + eroare
        start_p = coin_prices[0]
        end_p = coin_prices[-1]
        pct_ch = ((end_p - start_p) / (start_p + 1e-9)) * 100
        ups = sum(coin_directions)
        downs = HORIZON - ups

        summary_results.append({
            'coin_id': coin,
            'start_usd': round(start_p, 6),
            'end_usd': round(end_p, 6),
            'change_%': round(pct_ch, 2),
            'mlp_up_hours': ups,
            'mlp_down_hours': downs,
            'mlp_train_accuracy': round(mlp_train_acc, 4)
        })

        # Eroare: comparam predictia XGBoost vs date reale (pe training - ultimele 24h)
        if len(train) >= 48:
            last_24 = train.tail(24).copy()
            X_test = last_24[X_cols]
            y_real_ret = last_24['pct_return'].values
            y_real_dir = last_24['direction'].values
            y_pred_ret = model_xgb.predict(X_test)

            mae_return = np.mean(np.abs(y_real_ret - y_pred_ret))
            rmse_return = np.sqrt(np.mean((y_real_ret - y_pred_ret)**2))

            if model_mlp is not None:
                X_test_sc = scaler.transform(X_test)
                y_pred_dir = model_mlp.predict(X_test_sc)
                dir_accuracy = np.mean(y_real_dir == y_pred_dir)
            else:
                dir_accuracy = 0.0

            error_matrix_rows.append({
                'coin_id': coin,
                'MAE_return': round(mae_return, 6),
                'RMSE_return': round(rmse_return, 6),
                'MLP_direction_accuracy': round(dir_accuracy, 4),
                'MLP_train_accuracy': round(mlp_train_acc, 4)
            })

    if not all_results: return

    # === SALVARE ===
    res_df = pd.DataFrame(all_results)
    res_df['d_str'] = res_df['timestamp_utc'].dt.strftime('%Y-%m-%d')

    print("\n>>> [3] Salvare fisiere zilnice...")
    final_cols = ['timestamp_utc','coin_id','price_usd','total_volume_usd','hourly_increase','price-pop']
    for dv, data in res_df.groupby('d_str'):
        path = os.path.join(OUTPUT_DIR, f"{dv}.csv")
        data[final_cols].to_csv(path, index=False)
        print(f"    Salvat: {dv}.csv")

    # Sumar
    pd.DataFrame(summary_results).to_csv(os.path.join(OUTPUT_DIR, "prediction_summary.csv"), index=False)
    print(f"\n>>> [4] prediction_summary.csv salvat.")

    # === MATRICEA DE ERORI ===
    if error_matrix_rows:
        err_df = pd.DataFrame(error_matrix_rows)
        err_path = os.path.join(OUTPUT_DIR, "error_matrix.csv")
        err_df.to_csv(err_path, index=False)

        print(f"\n>>> [5] MATRICEA DE ERORI (error_matrix.csv):")
        print(f"{'coin_id':45s} {'MAE_ret':>10s} {'RMSE_ret':>10s} {'MLP_dir%':>10s} {'MLP_trn%':>10s}")
        print("-"*90)
        for _, r in err_df.iterrows():
            print(f"{r['coin_id']:45s} {r['MAE_return']:10.6f} {r['RMSE_return']:10.6f} {r['MLP_direction_accuracy']:10.4f} {r['MLP_train_accuracy']:10.4f}")
        print(f"\n    Medii globale:")
        print(f"      MAE return:            {err_df['MAE_return'].mean():.6f}")
        print(f"      RMSE return:           {err_df['RMSE_return'].mean():.6f}")
        print(f"      MLP directie accuracy: {err_df['MLP_direction_accuracy'].mean():.4f}")
        print(f"      MLP train accuracy:    {err_df['MLP_train_accuracy'].mean():.4f}")

if __name__ == "__main__":
    run_prediction()
