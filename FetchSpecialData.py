import pandas as pd
import san
import sys
import time
import os
san.ApiConfig.api_key = 'yvib2jijbrcivyfh_uw25dl5mk4qch5mf'

COINGECKO_TO_SANTIMENT = {
    "bitcoin": "bitcoin", "ethereum": "ethereum", "tether": "tether",
    "ripple": "ripple", "binancecoin": "binance-coin", "usd-coin": "usd-coin",
    "solana": "solana", "tron": "tron", "dogecoin": "dogecoin",
    "cardano": "cardano", "bitcoin-cash": "bitcoin-cash",
    "hyperliquid": "hyperliquid", "leo-token": "unus-sed-leo",
    "chainlink": "chainlink", "monero": "monero", "stellar": "stellar",
    "dai": "multi-collateral-dai", "litecoin": "litecoin",
    "avalanche-2": "avalanche", "hedera-hashgraph": "hedera-hashgraph",
    "sui": "sui", "shiba-inu": "shiba-inu", "polkadot": "polkadot-new",
    "uniswap": "uniswap", "near": "near-protocol", "aave": "aave",
    "bittensor": "bittensor", "okb": "okb", "zcash": "zcash",
    "ethereum-classic": "ethereum-classic", "mantle": "mantle",
    "pax-gold": "pax-gold", "tether-gold": "tether-gold",
    "internet-computer": "internet-computer",
    "crypto-com-chain": "crypto-com-coin", "paypal-usd": "paypal-usd",
    "kaspa": "kaspa", "pepe": "pepe", "aptos": "aptos",
    "ondo-finance": "ondo-finance", "ethena-usde": "ethena-usde",
    "jito-staked-sol": "jito-staked-sol", "usds": "usds",
    "weth": "weth", "wrapped-eeth": "wrapped-eeth",
}

RATE_LIMIT_WAIT = 1860

def get_social_volume(coin_id, date_str, slug):
    s_date = f"{date_str}T00:00:00Z"
    e_date = f"{date_str}T23:59:59Z"
    try:
        data = san.get(f"social_volume_total/{slug}",
                       s_date=s_date, e_date=e_date, interval="1d")
        if not data.empty and 'value' in data.columns:
            return float(data['value'].iloc[0]), False
        return None, False
    except Exception as e:
        err_str = str(e)
        if "429" in err_str or "Rate Limit" in err_str:
            return None, True
        print(f"    [ERROR] {coin_id} ({slug}): {err_str[:100]}")
        return None, False
    finally:
        time.sleep(1.5)

def main():
    if len(sys.argv) < 2:
        print("Usage: python FetchSpecialData.py <YYYY-MM-DD>")
        sys.exit(1)

    target_date = sys.argv[1]
    csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "historical_hourly", f"{target_date}.csv")
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: {csv_path} not found"); sys.exit(1)

    sv_col = f"social_volume_{target_date.replace('-', '_')}"
    if sv_col in df.columns:
        non_null = df[sv_col].dropna()
        if len(non_null) > 0 and (non_null > 0).any():
            print(f"[SKIP] {target_date} already has social data")
            return

    print(f"Loaded: {csv_path}")
    coins = df['coin_id'].unique()
    print(f"Found {len(coins)} coins")

    sv_data = {}
    for i, coin in enumerate(coins, 1):
        slug = COINGECKO_TO_SANTIMENT.get(coin)
        if not slug:
            print(f"[{i}/{len(coins)}] {coin} -> [SKIP]"); continue

        sv, rate_limited = get_social_volume(coin, target_date, slug)
        if rate_limited:
            print(f"[{i}/{len(coins)}] {coin} -> [RATE LIMITED] Waiting {RATE_LIMIT_WAIT//60}min...")
            time.sleep(RATE_LIMIT_WAIT)
            sv, rate_limited = get_social_volume(coin, target_date, slug)
            if rate_limited:
                print("  Still limited. Saving and stopping."); break

        if sv is not None:
            sv_data[coin] = sv
            print(f"[{i}/{len(coins)}] {coin} -> {slug} sv={sv}")
        else:
            print(f"[{i}/{len(coins)}] {coin} -> {slug} [NO DATA]")

    df[sv_col] = df['coin_id'].map(sv_data)
    df.to_csv(csv_path, index=False)
    print(f"Done: {len(sv_data)} found. Saved: {csv_path}")

if __name__ == "__main__":
    main()
