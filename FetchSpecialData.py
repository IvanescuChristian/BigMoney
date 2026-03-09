import pandas as pd
import san
import sys
from datetime import datetime, timedelta
import time
import os
san.ApiConfig.api_key = 'yvib2jijbrcivyfh_uw25dl5mk4qch5mf' # nu i dau hash,e ok


def get_coin_slugs():
    try:
        projects = san.get("projects/all")
        coin_slug_map = {}
        for _, p in projects.iterrows():
            if pd.notna(p['name']):
                coin_slug_map[p['name'].lower()] = p['slug']
            if pd.notna(p['ticker']):
                coin_slug_map[p['ticker'].lower()] = p['slug']
        print(f"Successfully fetched {len(coin_slug_map)} coin slugs.")
        return coin_slug_map
    except Exception as e:
        print(f"Error on coin : {e}")
        return {}

def get_soc_vol_c(coin_name, date_str, coin_slug_map):
    processed_coin_name = coin_name.replace("-"," ")
    coin_slug = coin_slug_map.get(processed_coin_name.lower())
    if not coin_slug:
        print(f"Warning: Could not find Santiment slug for coin '{coin_name}' (processed as '{processed_coin_name}'). Skipping social volume fetch for this coin.")
        return None
    s_date = f"{date_str}T00:00:00Z"#day start(not rly)
    e_date = f"{date_str}T23:59:59Z"

    print(f"Attempting to fetch social volume for {coin_name} (slug: {coin_slug}) on {date_str}...")
    try:
        data = san.get(f"social_volume_total/{coin_slug}",s_date=s_date,e_date=e_date,interval="1d")
        if not data.empty and 'value' in data.columns:
            social_volume = data['value'].iloc[0]
            print(f"Successfully fetched social volume for {coin_name}: {social_volume}")
            return social_volume
        else:
            print(f"No sv data found for {coin_name} on {date_str}")
            return None
    except Exception as e:
        print(f"Error geting sv for {coin_name} on {date_str}: {e}")
        return None
    finally:
        time.sleep(1)

def main():
    if len(sys.argv) < 2:
        print("Usage: python your_script_name.py <YYYY-MM-DD>")
        sys.exit(1)
    target_date_str = sys.argv[1]
    csv_file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),"historical_hourly",f"{target_date_str}.csv")
    try:
        df = pd.read_csv(csv_file_name)
        print(f"Successfully loaded CSV file: '{csv_file_name}'")
    except FileNotFoundError:
        print(f"Error: CSV file '{csv_file_name}' lost(never was). If u may, check the path.")
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred while reading CSV file '{csv_file_name}': {e}")
        sys.exit(1)
    if 'coin_id' not in df.columns:
        print("Error: The CSV file must contain a column named 'coin_id' with cryptocurrency identifiers.")
        sys.exit(1)
    distinct_coins = df['coin_id'].unique()
    print(f"Found distinct coins in '{csv_file_name}': {distinct_coins}")
    coin_slug_map = get_coin_slugs()
    if not coin_slug_map:
        print("Failed to retrieve coin slug mapping from Santiment API. Please check your API key and network connection.")
        sys.exit(1)
    sv_data = {}
    for coin in distinct_coins:
        volume = get_soc_vol_c(coin,target_date_str,coin_slug_map)
        sv_data[coin] = volume
    new_c_name = f"social_volume_{target_date_str.replace('-','_')}"
    df[new_c_name] = None
    for index, row in df.iterrows():
        coin = row['coin_id']
        if coin in sv_data:
            df.at[index,new_c_name] = sv_data[coin]
        else:
            df.at[index,new_c_name] = None
            print(f"Note: No social volume data available for '{coin}'. Column value set to None.")
    try:
        df.to_csv(csv_file_name,index=False)
        print(f"Succesfuly updated and saved '{csv_file_name}'.")
    except Exception as e:
        print(f"Error : {e}")

if __name__ == "__main__":
    main()