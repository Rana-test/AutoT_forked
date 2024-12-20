import logging
from logging.handlers import RotatingFileHandler
from api_helper import ShoonyaApiPy
import os
import pyotp
import pandas as pd
from datetime import datetime, time, timezone
import time as sleep_time
### Add condition for exception, when index moves way beyond adjustment
import yaml
import helpers.helper as h
import pytz
import requests
import sys

if os.path.exists('helpers/creds.yml'):
    sys.path.append('/home/rana/trading/newAuto/AutoT/env/lib/python3.10/site-packages')

import upstox_client
from upstox_client.rest import ApiException

def load_yaml_to_globals(yaml_file):
    with open(yaml_file, 'r') as file:
        data = yaml.safe_load(file)  # Load YAML content
        globals().update(data)

def save_globals_to_yaml(yaml_file, global_vars):
    # Filter out non-serializable objects if needed
    keys=['BankExpiry','BankSymbol','EntryType','Entry_Date','Expiry','ICT','IFT',
          'Past_M2M','Symbol','Update_EOD','counter_test'
          ,'enter_strategy','enter_today','interval','live','lot_size','lots','nfo_sym_path',
          'nse_sym_path','num_adjustments','percent_of_max_profit','positions_data',
          'stop_loss','target_profit', 'access_token','instrument_csv_path', 'log_file_path']
    serializable_vars = {key: value for key, value in global_vars.items() if key in keys}
    with open(yaml_file, 'w') as file:
        yaml.dump(serializable_vars, file, default_flow_style=False)


def is_within_timeframe(start, end):
    now = datetime.now(timezone.utc) 
    start_time = now.replace(hour=int(start.split(':')[0]), minute=int(start.split(':')[1]), second=0, microsecond=0)
    end_time = now.replace(hour=int(end.split(':')[0]), minute=int(end.split(':')[1]), second=0, microsecond=0)
    return start_time <= now <= end_time

def identify_session():
    # IST to UTC
    # 8:30 == 3:00
    # 9:00 ==  3:30
    # 9:15 == 3:45
    # 12:25 == 6:55
    # 12:27 == 6:57
    # 12:30 == 7:00
    # 15:30 == 10:00

    if is_within_timeframe("03:00", "06:55"):
        return {"session": "session1", "start_time": "03:45", "end_time": "06:57"}
    elif is_within_timeframe("07:00", "20:00"):
        return {"session": "session2","start_time": "07:00", "end_time": "20:00"}
    return None

def main():
    global nifty_nse_token, email_subject,future_price_token, nifty_bank_nse_token
    global future_price_bnk_token, access_token

    session = identify_session()
    if not session:
        print("No active trading session.")
        return
    
    # Load all environment variables
    load_yaml_to_globals('config_upstox_v1.yml')
    
    email_subject ="|Initializing...|"

    # Get all global variables and their values, excluding built-in ones
    global_vars = {key: value for key, value in globals().items() if not key.startswith('__')}

    instrument_df = None
    for i in range(10):
        try:
            instrument_df = pd.read_csv(global_vars.get("instrument_csv_path"))
        except:
            i+=1
            sleep_time.sleep(10)
            continue

        if instrument_df is not None:
            break

    # Initialize logger
    logger = h.upstox_logger_init()
    
    configuration = upstox_client.Configuration()
    # Login
    try:
        access_token= global_vars.get("access_token")
        configuration.access_token = access_token   
        market_quote_api_instance = upstox_client.MarketQuoteApi(upstox_client.ApiClient(configuration))
        current_strike = market_quote_api_instance.ltp("NSE_INDEX|Nifty 50","2.0")
    except:
        access_token = h.upstox_login(logger)
        configuration.access_token = access_token   
        market_quote_api_instance = upstox_client.MarketQuoteApi(upstox_client.ApiClient(configuration))
        if os.path.exists('config_upstox_v1.yml'):
            save_globals_to_yaml("config_upstox_v1.yml", global_vars)

    while is_within_timeframe("03:00", "03:45"):
        sleep_time.sleep(60)

    counter=0

    # Start Monitoring
    while is_within_timeframe(session.get('start_time'), session.get('end_time')):
        # Flush logs
        open(global_vars.get('log_file_path'), 'w').close()
        logger.info(f"SESSION: {session}")
        logger.info(f"### Day Count: {h.count_working_days(global_vars)} ###")

        email_subject=""

        CM2M = global_vars.get("Past_M2M") 

        for pos in os.listdir("Upstox_Positions"): #+["nifty"]:
            # Get Current Positions
            if pos.split('.')[-1] =='csv':
                open_positions= pd.read_csv('Upstox_Positions/'+pos)
                IC_delta_threshold = 40
                IF_delta_threshold = 50
            else:
                open_positions=None
                IC_delta_threshold = global_vars.get("ICT")
                IF_delta_threshold = global_vars.get("IFT")
            positions_df, m2m, closed_m2m = h.upstox_get_current_positions(market_quote_api_instance, logger, instrument_df, global_vars, open_positions)
            symbol=None
            expiry=None
            minsp=None
            maxsp=None
            if pos.split("_")[0]=="nifty":
                current_strike = float(market_quote_api_instance.ltp("NSE_INDEX|Nifty 50","2.0").data["NSE_INDEX:Nifty 50"].last_price)
                email_head = "<NIFTY>"
                symbol="NIFTY"
                expiry=global_vars.get('Expiry')
                # minsp=22000
                # maxsp=27000
            elif pos.split("_")[0]=="banknifty":
                current_strike = float(market_quote_api_instance.ltp("NSE_INDEX|Nifty Bank","2.0").data["NSE_INDEX:Nifty Bank"].last_price)
                email_head = "<BANKNIFTY>"
                symbol="BANKNIFTY"
                expiry=global_vars.get('BankExpiry')
                # minsp=48000
                # maxsp=58000
            email_sub = monitor_trade(logger, access_token, global_vars,positions_df, m2m, closed_m2m, current_strike, symbol, expiry, minsp,maxsp, IC_delta_threshold, IF_delta_threshold)
            CM2M+=int(m2m)
            email_subject += email_head + email_sub +"|"
            print(email_subject)    
        exit(0)
        sleep_time.sleep(global_vars.get("interval"))
        if counter % 10 == 0:
            h.send_email(f"CM2M:{CM2M}"+ email_subject, global_vars)
        counter+=1

    # Update Past_M2M at EOD
        # if session ==2 and tod = eod: update config
        #     if config['Update_EOD']==0:
        #         # Update Settled Amount
        #         config['Update_EOD']=1
        #         config['Past_M2M']=config['Past_M2M']+closed_m2m
        #         exit(0)
        # else:
        #     config['Update_EOD']=0
        # save_config()
    
    # Logout
    api.logout()
    global_vars['counter_test']=counter_test
    save_globals_to_yaml('config_v2.yml', global_vars)

def monitor_trade(logger, access_token, global_vars, positions_df, m2m, closed_m2m , current_strike, symbol, expiry, minsp,maxsp, IC_delta_threshold, IF_delta_threshold):

    # If no positions found then check if entry needs to be made and create entry
    # if (positions_df is None or positions_df.empty):
    #     if global_vars.get('enter_today'):
    #         email_subject ="|Entering Tade|"
    #         if global_vars.get('enter_strategy') =='IF':
    #             h.enter_trade_ironfly(api, logger, global_vars)
    #         elif global_vars.get('enter_strategy') =='M':
    #             h.enter_trade_manual(api, logger, global_vars)
    #         # elif global_vars.get('enter_strategy') =='IC_3SD':
    #     else:
    #         email_subject ="|No Positions found|"
    
    # Get breakevens
    lower_be, higher_be = h.upstox_calculate_breakevens(positions_df, global_vars)
    logger.info(f"LOWER BE: {lower_be}, HIGHER BE: {higher_be}")
    
    # Step 2: Check adjustment signal

    delta, pltp, cltp, profit_leg, loss_leg, strategy, pe_hedge_diff, ce_hedge_diff, current_strike, pstrike, cstrike = h.upstox_calculate_delta(positions_df, current_strike)
    logger.info(f"CURRENT STRIKE : {current_strike}")
    logger.info(f"STRATEGY : {strategy}")
    
    delta_threshold = IC_delta_threshold if strategy=="IC" else IF_delta_threshold
    logger.info(f"DELTA : {delta} | DELTA_THRESHOLD: {delta_threshold}")
    # try:
    #     h.calculate_metrics(logger, positions_df)
    # except Exception as e:
    #     print(e)
    email_subject = f'DELTA: {delta}% | M2M: {m2m} | SP: {current_strike} | Strategy: {strategy}'

    # Calculate max_profit and exit if condition met
    max_profit = float((positions_df['qty'].astype(int)*positions_df['netupldprc'].astype(float)).sum()) * -1 + closed_m2m + global_vars.get('Past_M2M')
    
    # if h.auto_exit(logger, api, global_vars, max_profit, strategy, m2m, positions_df):
    #     email_subject = f'TARGET PROFIT/LOSS HIT. Exiting..'

    # adj_order, new_delta = h.require_adjustments(logger, api, global_vars, strategy, delta, positions_df, profit_leg, loss_leg, pltp, cltp, symbol, expiry, minsp,maxsp, IC_delta_threshold, IF_delta_threshold )

    # if adj_order is not None:
    #     logger.info("<<<<<PERFORM ADJUSTMENTS>>>>>")
    #     adj_order['qty']= adj_order['qty'].apply(lambda x: abs(int(x)))
    #     logger.info(adj_order)
    #     email_subject = f'ADJUSTMENT MADE | PROB NEW DELTA: {new_delta}%'
    #     # place adjustment orders
    #     h.execute_basket(logger, global_vars, api,adj_order)
    #     h.send_email(email_subject, global_vars)
    # else:
    #     logger.info("<<<<<NO ADJUSTMENTS NEEDED>>>>>")

    return email_subject


def fetch_ltp(instrument_key, ACCESS_TOKEN):
    url = 'https://api.upstox.com/v2/quote'
    headers = {
        'Authorization': f'Bearer {ACCESS_TOKEN}',
        'Accept': 'application/json'
    }
   
    response = requests.get(url, params={'instrument_key': instrument_key}, headers=headers)
    return response.json()

def is_within_time_range():
    # Get the current UTC time
    now = datetime.now(pytz.utc).time()

    # Define the start and end times
    start_time = time(3, 45)  # 3:45 a.m. UTC
    end_time = time(21, 00)   # 10:00 a.m. UTC

    # Check if the current time is within the range
    return start_time <= now <= end_time

if __name__ =="__main__":
    main()
    
    