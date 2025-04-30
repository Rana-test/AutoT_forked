import pandas as pd
import numpy as np
from api_helper import get_time
from datetime import datetime, timedelta, time
import math
import calendar
import os
from datetime import datetime, timedelta
import logging
import time as sleep_time
import ta
from zoneinfo import ZoneInfo 
logging.basicConfig(level=logging.INFO)
################# Helper functions #################

month_mapping = {
    '1': 'JAN', '2': 'FEB', '3': 'MAR', '4': 'APR', '5': 'MAY', '6': 'JUN',
    '7': 'JUL', '8': 'AUG', '9': 'SEP', 'O': 'OCT', 'N': 'NOV', 'D': 'DEC'
}

holiday_dict ={
    '2025-05-01':'2025-04-30',
    '2025-10-02':'2025-10-01',
    '2025-12-25':'2025-12-24',
}


def get_india_vix(api):
    return round(float(api.get_quotes(exchange="NSE", token=str(26017))['lp']),2)

def find_last_thursday(year, month):
    """Finds the last Thursday of a given month and year."""
    last_day = datetime(year, month, 1).replace(day=calendar.monthrange(year, month)[1])
    while last_day.weekday() != 3:  # Thursday is weekday 3
        last_day -= timedelta(days=1)
    return last_day.day

def get_next_thursday_between_4_and_12_days(now):
    today = now.date()

    for i in range(5, 12):  # Start from 5 (to be > 4) up to 11 (< 12)
        target_date = today + timedelta(days=i)
        if target_date.weekday() == 3:  # 0=Monday, ..., 3=Thursday
            return target_date.strftime('%Y-%m-%d')
    
    return None  # If no Thursday found in range

def convert_option_symbol(input_symbol):
    # Ensure input is valid
    if not isinstance(input_symbol, str) or len(input_symbol) < 12:
        raise ValueError("Invalid symbol format. Expected format: NIFTY2530622300PE or NIFTY25FEB22450PE")

    # Extract the underlying index (e.g., NIFTY)
    index = input_symbol[:5]

    # Extract the year
    year_prefix = input_symbol[5:7]

    # Determine if it's a weekly or monthly expiry
    remaining_part = input_symbol[7:-2]  # Excludes index, year, and option type

    # Check if the month part is in the three-letter format (e.g., FEB) for monthly options
    if remaining_part[:3].isalpha():
        # Monthly expiry format: NIFTY25FEB22450PE
        month_abbr = remaining_part[:3].upper()
        strike_price = remaining_part[3:]
        expiry_day = None  # Monthly expiry day to be determined separately
    else:
        # Weekly expiry format: NIFTY2530622300PE (where "306" means 6th March)
        month_code = remaining_part[0]  # Could be a single letter (O, N, D) or a digit (1-9)
        day = remaining_part[1:3]  # Extract the day (e.g., '06')
        strike_price = remaining_part[3:]

        # Convert month code to full month abbreviation
        if month_code in month_mapping:
            month_abbr = month_mapping[month_code]
        else:
            raise ValueError(f"Invalid month code: {month_code}")

        expiry_day = day  # Weekly expiry has an explicit day

    # Determine the final expiry format
    if expiry_day:
        # Weekly expiry format: NIFTY06MAR25P22300
        new_symbol = f"{index}{expiry_day}{month_abbr}{year_prefix}{'P' if input_symbol[-2:] == 'PE' else 'C'}{strike_price}"
    else:
        # Monthly expiry format: Find the last Thursday of the month
        expiry_year = int(f"20{year_prefix}")
        expiry_month = datetime.strptime(month_abbr, "%b").month
        expiry_day = find_last_thursday(expiry_year, expiry_month)  # Function to get last Thursday

        new_symbol = f"{index}{expiry_day}{month_abbr}{year_prefix}{'P' if input_symbol[-2:] == 'PE' else 'C'}{strike_price}"

    return new_symbol

def get_option_chain(upstox_opt_api, instrument, expiry):
    # option_chain = upstox_opt_api.get_option_chain(symbol=instrument, expiry_date=expiry)
    option_chain = upstox_opt_api.get_put_call_option_chain(instrument_key=instrument,expiry_date=expiry)
    return option_chain.data

def get_nearest_delta_options(option_chain_data, upstox_instruments, delta):
    call_symbol = None
    put_symbol = None
    min_call_diff = float("inf")
    min_put_diff = float("inf")

    for option in option_chain_data:
        # Access call and put options using dot notation
        call_option = option.call_options
        put_option = option.put_options

        # Process call options
        if call_option and call_option.option_greeks:
            call_symb = upstox_instruments[upstox_instruments['instrument_key']==call_option.instrument_key].tradingsymbol.values[0]
            call_delta = call_option.option_greeks.delta
            call_oi = float(call_option.market_data.oi)
            if call_delta is not None and abs(call_delta - delta) < min_call_diff and call_symb[-4:-2] == '00' and call_oi>10000:
                min_call_diff = abs(call_delta - delta)
                call_symbol = call_option.instrument_key
                upstox_ce_ltp = call_option.market_data.ltp
                co = call_option

        # Process put options
        if put_option and put_option.option_greeks:
            put_symb = upstox_instruments[upstox_instruments['instrument_key']==put_option.instrument_key].tradingsymbol.values[0]
            put_delta = put_option.option_greeks.delta
            put_oi = float(put_option.market_data.oi) if put_option.market_data.oi is not None else 0
            if put_delta is not None and abs(put_delta + delta) < min_put_diff and put_symb[-4:-2] == '00' and put_oi>10000:
                min_put_diff = abs(put_delta + delta)
                put_symbol = put_option.instrument_key
                upstox_pe_ltp = put_option.market_data.ltp
                po = put_option

    call_bid = float(co.market_data.bid_price)
    call_ask = float(co.market_data.ask_price)
    call_bid_ask = call_ask-call_bid
    put_bid = float(po.market_data.bid_price)
    put_ask = float(po.market_data.ask_price)
    put_bid_ask = put_ask-put_bid
    return call_symbol, put_symbol, upstox_ce_ltp, upstox_pe_ltp, po, co , co.market_data.oi, po.market_data.oi, co.option_greeks.delta, po.option_greeks.delta, call_bid_ask, put_bid_ask

def get_positions(upstox_opt_api, finvasia_api, instrument, expiry,trade_qty,upstox_instruments, delta):
    SPAN_Expiry = datetime.strptime(expiry, "%Y-%m-%d").strftime("%d-%b-%Y").upper()
    trade_details={}
    option_chain = get_option_chain(upstox_opt_api, instrument, expiry)
    upstox_ce_instrument_key, upstox_pe_instrument_key, upstox_ce_ltp, upstox_pe_ltp,  po, co, call_oi, put_oi, call_delta, put_delta, call_bid_ask, put_bid_ask = get_nearest_delta_options(option_chain,upstox_instruments,delta)
    trade_details['upstox_ce_instrument_key']=upstox_ce_instrument_key
    trade_details['upstox_pe_instrument_key']=upstox_pe_instrument_key
    trade_details['call_oi']=call_oi
    trade_details['put_oi']=put_oi
    trade_details['call_delta']=call_delta*100
    trade_details['put_delta']=put_delta*100
    trade_details['call_bid_ask']=call_bid_ask
    trade_details['put_bid_ask']=put_bid_ask
    upstox_ce_symbol = upstox_instruments[upstox_instruments['instrument_key']==upstox_ce_instrument_key]['tradingsymbol'].values[0]
    # trade_details['upstox_ce'] = upstox_ce_symbol
    upstox_pe_symbol = upstox_instruments[upstox_instruments['instrument_key']==upstox_pe_instrument_key]['tradingsymbol'].values[0]
    # trade_details['upstox_pe'] = upstox_pe_symbol
    instruments = []
    instruments.append({"instrument_key": upstox_ce_instrument_key, "quantity": trade_qty, "transaction_type": "SELL", "product": "D", "price": upstox_ce_ltp})
    instruments.append({"instrument_key": upstox_pe_instrument_key, "quantity": trade_qty, "transaction_type": "SELL", "product": "D", "price": upstox_pe_ltp})
    current_index_price = round(float(finvasia_api.get_quotes(exchange="NSE", token=str(26000))['lp']),2)
    atm_iv =0
    strike_interval = 100
    remainder = math.fmod(current_index_price, strike_interval)
    if remainder > strike_interval / 2:
        atm_strike = math.ceil(current_index_price / strike_interval) * strike_interval
    else:
        atm_strike = math.floor(current_index_price / strike_interval) * strike_interval
    for sp in option_chain:
        if sp.strike_price == atm_strike:
            atm_iv = (float(sp.call_options.option_greeks.iv)+float(sp.put_options.option_greeks.iv))/2

    trade_details['current_index_price']=current_index_price
    lower_range = round((int(upstox_pe_symbol[-7:-2])-current_index_price)/current_index_price*100,2)
    trade_details['lower_range']=lower_range
    upper_range = round((int(upstox_ce_symbol[-7:-2])-current_index_price)/current_index_price*100,2)
    trade_details['upper_range']=upper_range
    trading_range = int(upstox_ce_symbol[-7:-2])-int(upstox_pe_symbol[-7:-2])
    trade_details['trading_range']=trading_range//2
    trade_details['range_per']=round(trading_range/current_index_price*100,2)

    # Build instruments for finvasia
    fin_pe_symbol = convert_option_symbol(upstox_pe_symbol)
    fin_ce_symbol = convert_option_symbol(upstox_ce_symbol)
    trade_details['fin_pe_symbol']=fin_pe_symbol
    trade_details['fin_ce_symbol']=fin_ce_symbol

    span_res = finvasia_api.span_calculator('FA417461',[
            {"prd":"M","exch":"NFO","instname":"OPTSTK","symname":"NIFTY","exd":SPAN_Expiry,"optt":"PE","strprc":str(fin_pe_symbol[-5:])+".00","buyqty":"0","sellqty":str(trade_qty),"netqty":"0"},
            {"prd":"M","exch":"NFO","instname":"OPTSTK","symname":"NIFTY","exd":SPAN_Expiry,"optt":"CE","strprc":str(fin_ce_symbol[-5:])+".00","buyqty":"0","sellqty":str(trade_qty),"netqty":"0"}
        ])

    trade_margin=float(span_res['span_trade']) + float(span_res['expo_trade'])
    trade_details['trade_margin']=trade_margin
    finvasia_pe_ltp=float(finvasia_api.get_quotes(exchange="NFO", token=str(fin_pe_symbol))['lp'])
    trade_details['finvasia_pe_ltp']=finvasia_pe_ltp
    finvasia_ce_ltp=float(finvasia_api.get_quotes(exchange="NFO", token=str(fin_ce_symbol))['lp'])
    trade_details['finvasia_ce_ltp']=finvasia_ce_ltp
    tot_fin_premium = round(trade_qty*(finvasia_pe_ltp+finvasia_ce_ltp),2)
    trade_details['tot_fin_premium']=tot_fin_premium
    fin_mtm_per= round(tot_fin_premium/trade_margin*100,2)
    trade_details['fin_mtm_per']=str(fin_mtm_per)+"%"
    trade_details['INDIA_VIX']=get_india_vix(finvasia_api)
    try:
        # trade_details['INDIA_VIX_RSI']=get_nifty_rsi()
        trade_details['ATM_IV']=atm_iv
    except:
        trade_details['INDIA_VIX_RSI']=-1
        trade_details['ATM_IV']=-1
    return trade_details

################# Logic functions #################

def place_order(api, live, trading_symbol, buy_sell, qty, order_type):
    logging.info(f"Within place order")
    quantity = qty
    tradingsymbol= trading_symbol
    prd_type = 'M'
    exchange = 'NFO' 
    # disclosed_qty= 0
    price_type = 'MKT'
    price=0
    trigger_price = None
    retention='DAY'
    email_body = ''
    if live:
        logging.info(f"Placing order: {trading_symbol}, {buy_sell}, {qty}, {order_type}")
        response = api.place_order(buy_or_sell=buy_sell, product_type=prd_type, exchange=exchange, tradingsymbol=tradingsymbol, quantity=quantity, discloseqty=quantity,price_type=price_type, price=price,trigger_price=trigger_price, retention=retention, remarks=order_type)
        if response is None or 'norenordno' not in response:
            logging.info(f"None Response")
            return False, {'subject': "Order Placement Failed", 'body': "Order Placement Failed"}
        order_id = response['norenordno']
        logging.info(f"Order_id: {order_id}")
        email_body = f"Order placed successfully : Order No: {order_id}/n"
        email_body += f'buy_or_sell={buy_sell}, product_type={prd_type}, exchange={exchange}, tradingsymbol={tradingsymbol}, quantity={quantity}, discloseqty={quantity},price_type={price_type}, price={price},trigger_price={trigger_price}, retention={retention}, remarks={order_type}'
        for _ in range(10):  
            logging.info(f"Waiting for order execution confirmation")# Try for ~10 seconds
            sleep_time.sleep(1)
            orders = api.get_order_book()
            if orders:
                matching_orders = [o for o in orders if o['norenordno'] == order_id]
                if matching_orders:
                    order = matching_orders[0]
                    logging.info(f"Matching Order: {order}")
                    status = order['status']
                    logging.info(f"Order response: {status}")
                    if status == 'COMPLETE':
                        # subject = f"Order executed successfully.: Order No: {order_id}"
                        email_body+=f"Order executed successfully.: Order No: {order_id}"
                        return True, {'subject': "Order executed successfully", 'body': email_body}
                    elif status in ['REJECTED', 'CANCELLED']:
                        email_body+=f"Order {status}. Reason: {order.get('rejreason', 'Not available')}"
                        return False, {'subject': f"ORDER REJECTED : Reason: {order.get('rejreason', 'Not available')}", 'body': email_body}
            else:
                email_body = email_body+ f"Could not fetch order book./n"

        email_body = email_body+ "Timed out waiting for order update."
        logging.info(f"Order Execution Timed out")
        return True, {'subject': "Order Timed out", 'body': email_body}

    else:
        print(f'buy_or_sell={buy_sell}, product_type={prd_type}, exchange={exchange}, tradingsymbol={tradingsymbol}, quantity={quantity}, discloseqty={quantity},price_type={price_type}, price={price},trigger_price={trigger_price}, retention={retention}, remarks={order_type}')
        subject = f"{order_type} order for {tradingsymbol}"
        email_body = f'buy_or_sell={buy_sell}, product_type={prd_type}, exchange={exchange}, tradingsymbol={tradingsymbol}, quantity={quantity}, discloseqty={quantity},price_type={price_type}, price={price},trigger_price={trigger_price}, retention={retention}, remarks={order_type}'
        # send_email_plain(subject, email_body)
        logging.info(f"Dummy order placed: {email_body}")
        return True, {'subject': subject, 'body': email_body}


def get_minute_data(api, now=None):
    nifty_token = '26000'  # NSE|26000 is the Nifty 50 index
    
    # Define trading hours
    market_open = time(9, 15)
    market_close = time(15, 30)
    
    # Set current time if not provided
    if now is None:
        now = datetime.now(ZoneInfo("Asia/Kolkata"))
    
    # Adjust latest_time to the most recent trading minute
    def adjust_to_trading_hours(dt):
        dt = dt.astimezone(ZoneInfo("Asia/Kolkata")) 
        dt_time = dt.time()
        dt_date = dt.date()
        
        if dt_time > market_close:
            # After market close, use 15:30:00 of the same day
            return datetime.combine(dt_date, market_close, ZoneInfo("Asia/Kolkata"))
        elif dt_time < market_open:
            # Before market open, use 15:30:00 of the previous trading day
            prev_day = dt_date - timedelta(days=1)
            # Check if previous day is a weekday (Monday to Friday)
            while prev_day.weekday() >= 5:  # Skip Saturday (5) and Sunday (6)
                prev_day -= timedelta(days=1)
            return datetime.combine(prev_day, market_close, ZoneInfo("Asia/Kolkata"))
        else:
            # Within trading hours, round down to the nearest minute
            return dt.replace(second=0, microsecond=0)
    
    latest_time = adjust_to_trading_hours(now)
    
    # Time 90 days ago
    start_time = latest_time - timedelta(days=30)
    
    # Desired time format
    fmt = "%d-%m-%Y %H:%M:%S"
    
    # Convert times to seconds (assuming get_time is defined elsewhere)
    start_secs = get_time(start_time.strftime(fmt))  # dd-mm-YYYY HH:MM:SS
    end_secs = get_time(latest_time.strftime(fmt))
    
    # Fetch 1-minute candle data from Finvasia API
    bars = api.get_time_price_series(
        exchange='NSE',
        token=nifty_token,
        starttime=int(start_secs),
        endtime=int(end_secs),
        interval=1  # 1-minute candles
    )
    
    # Create DataFrame
    df = pd.DataFrame.from_dict(bars)
    df.rename(columns={
        'into': 'open',
        'inth': 'high',
        'intl': 'low',
        'intc': 'close'
    }, inplace=True)
    
    # Convert time to datetime
    df['time'] = pd.to_datetime(df['time'], dayfirst=True)
    
    # Select and convert columns to float
    df = df[['time', 'open', 'high', 'low', 'close']]
    df[['open', 'high', 'low', 'close']] = df[['open', 'high', 'low', 'close']].astype(float)
    
    # Filter data to ensure it's within trading hours (9:15:00 to 15:30:00)
    df = df[(df['time'].dt.time >= market_open) & (df['time'].dt.time <= market_close)]
    # df = df.reset_index()

    return df

def calculate_rsi_wilder(close_prices, period=14):
    delta = close_prices.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    # Use exponential weighted mean for Wilder's smoothing
    avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_supertrend(df_minute):
    """
    Calculates Supertrend (10 ATR, 3.5 factor) on hourly Nifty 50 candles without talib.
    
    Returns:
        pd.DataFrame: Hourly candles with Supertrend values and signals
    """
   
    # Convert timestamp to datetime and set as index
    df = df_minute.copy()
    df['time'] = pd.to_datetime(df_minute['time'])
    df.set_index('time', inplace=True)


    ############################## START HOURLY #################
    # # Aggregate to hourly candles
    # df_hourly = df.resample('1h', label='right', closed='right').agg({
    #     'open': 'first',
    #     'high': 'max',
    #     'low': 'min',
    #     'close': 'last',
    #     # 'volume': 'sum'
    # }).dropna()

    ########################## END HOURLY #####################
    ########################## START MIN #####################
    # Sort descending (latest to earliest)
    df = df.sort_index(ascending=False)

    # Assign reverse-time bins: each 60-minute chunk gets a unique group number
    df['minutes_from_latest'] = ((df.index[0] - df.index).total_seconds() // 60).astype(int)
    df['reverse_hour_bin'] = (df['minutes_from_latest'] // 60).astype(int)

    # Aggregate into OHLC
    agg_funcs = {
        'open': 'last',   # Because we're going from latest to earliest
        'high': 'max',
        'low': 'min',
        'close': 'first'
    }

    df_hourly = df.groupby('reverse_hour_bin').agg(agg_funcs)

    # Optional: Add timestamp for the latest time in each bin
    df_hourly['time'] = df.groupby('reverse_hour_bin').apply(lambda x: x.index[0])

    # Reorder columns if needed
    df_hourly = df_hourly[['time', 'open', 'high', 'low', 'close']]
    df_hourly.set_index('time', inplace=True)
    df_hourly = df_hourly.sort_index(ascending=True)

    ###########################END MIN#########################
        
    # Calculate True Range (TR)
    df_hourly['tr'] = pd.concat([
        df_hourly['high'] - df_hourly['low'],
        (df_hourly['high'] - df_hourly['close'].shift(1)).abs(),
        (df_hourly['low'] - df_hourly['close'].shift(1)).abs()
    ], axis=1).max(axis=1)
    
    # Calculate ATR using EMA with alpha = 1/10 (RMA equivalent)
    Periods = 10
    df_hourly['atr'] = df_hourly['tr'].ewm(alpha=1/Periods, adjust=False).mean()
    
    # Initialize Supertrend lists
    up_list = [np.nan] * len(df_hourly)
    dn_list = [np.nan] * len(df_hourly)
    trend_list = [np.nan] * len(df_hourly)
    
    Multiplier = 3.5
    
    # Calculate Supertrend
    for i in range(len(df_hourly)):
        # src = (df_hourly['high'].iloc[i] + df_hourly['low'].iloc[i]) / 2
        src = df_hourly['close'].iloc[i]
        if i == 0:
            # First bar initialization
            up = src - Multiplier * df_hourly['atr'].iloc[i]
            dn = src + Multiplier * df_hourly['atr'].iloc[i]
            trend = 1  # Start with uptrend
        else:
            up_temp = src - Multiplier * df_hourly['atr'].iloc[i]
            dn_temp = src + Multiplier * df_hourly['atr'].iloc[i]
            up1 = up_list[i-1]
            dn1 = dn_list[i-1]
            previous_close = df_hourly['close'].iloc[i-1]
            
            # Adjust bands
            up = max(up_temp, up1) if previous_close > up1 else up_temp
            dn = min(dn_temp, dn1) if previous_close < dn1 else dn_temp
            
            # Determine trend
            if trend_list[i-1] == -1 and df_hourly['close'].iloc[i] > dn1:
                trend = 1
            elif trend_list[i-1] == 1 and df_hourly['close'].iloc[i] < up1:
                trend = -1
            else:
                trend = trend_list[i-1]
        
        up_list[i] = up
        dn_list[i] = dn
        trend_list[i] = trend
    
    # Add Supertrend to DataFrame
    df_hourly['up'] = up_list
    df_hourly['dn'] = dn_list
    df_hourly['trend'] = trend_list
    
    # Generate signals
    # df_hourly['buySignal'] = (df_hourly['trend'] == 1) & (df_hourly['trend'].shift(1) == -1)
    # df_hourly['sellSignal'] = (df_hourly['trend'] == -1) & (df_hourly['trend'].shift(1) == 1)

    # Calculate 20 and 50 EMA
    df_hourly['ema20'] = df_hourly['close'].ewm(span=20, adjust=False).mean()
    df_hourly['ema34'] = df_hourly['close'].ewm(span=34, adjust=False).mean()
    # df_hourly['adx'] = ta.trend.adx(high=df_hourly['high'], low=df_hourly['low'], close=df_hourly['close'], window=20)
    # Calculate RSI
    df_hourly['rsi'] = calculate_rsi_wilder(df_hourly['close'], period=14)
    df_hourly['entry_signal'] = 0
    # df_hourly.loc[(df_hourly['close'] < df_hourly['ema20']) & (df_hourly['trend'] == -1) & (df_hourly['adx'] > 25), 'entry_signal'] = 1
    # df_hourly.loc[(df_hourly['close'] > df_hourly['ema20']) & (df_hourly['trend'] == 1) & (df_hourly['adx'] > 25), 'entry_signal'] = -1
    df_hourly.loc[(df_hourly['close'] < df_hourly['ema20']) & (df_hourly['close'] < df_hourly['ema34']) & (df_hourly['trend'] == -1) & (df_hourly['rsi'] <50), 'entry_signal'] = 1
    df_hourly.loc[(df_hourly['close'] > df_hourly['ema20']) &(df_hourly['close'] > df_hourly['ema34']) & (df_hourly['trend'] == 1)& (df_hourly['rsi'] > 50) , 'entry_signal'] = -1
    # Initialize the exit_signal column to 0
    df_hourly['exit_signal'] = 0
    # Set exit_signal to 1 when the trend changes (current trend != previous trend)
    df_hourly.loc[df_hourly['trend'] != df_hourly['trend'].shift(1), 'exit_signal'] = 1
    
    return df_hourly

# Read trade history
trade_history_file='trade_history_stema.csv'
logging.info(f"Reading Trade History")
if os.path.exists(trade_history_file):
    trade_history = pd.read_csv(trade_history_file, parse_dates=['time', 'exit_timestamp'])
    # Close any trades whose expiry is past today's date
    for i, row in trade_history.iterrows():
        row_expiry = row['expiry']
        days_to_expiry = (datetime.strptime(row_expiry, "%Y-%m-%d").date() - datetime.now(ZoneInfo("Asia/Kolkata")).date()).days
        if days_to_expiry < 1:
            trade_history.at[i, 'status'] = 'CLOSED'
else:
    trade_history = pd.DataFrame(columns=['time', 'trading_symbol', 'expiry','order_type', 'order_action', 'order_leg', 'status','exit_timestamp'])

import math

def fixed_ratio_position_size(base_size, delta, total_profit):
    """
    Calculates position size allowing growth and reduction below initial size.

    Parameters:
    - base_size (int): Neutral starting size (e.g., 10 lots)
    - delta (float): Profit required to add/subtract a unit
    - total_profit (float): Cumulative profit or loss

    Returns:
    - int: Adjusted position size
    """
    if total_profit >= 0:
        adjustment = math.floor(math.sqrt(total_profit / delta))
    else:
        adjustment = -math.floor(math.sqrt(abs(total_profit) / delta))
    
    position_size = base_size + adjustment
    position_size = max(1, position_size)
    return position_size # Ensure at least 1 lot is used


def get_revised_qty_margin(orders, upstox_charge_api, min_coll):
    main_leg = orders['Main'] #={'trading_symbol':main_leg['fin_pe_symbol'], , 'trading_up_symbol':main_leg['upstox_pe_instrument_key'], 'order_action':'S', 'order_qty':str(trade_qty), 'order_type':'PUT'}
    hedge_leg = orders['Hedge'] #={'trading_symbol':hedge_leg['fin_pe_symbol'], , 'trading_up_symbol':main_leg['upstox_pe_instrument_key'], 'order_action':'B', 'order_qty':str(trade_qty), 'order_type':'PUT'}
    instruments = [
    {
        "instrument_key": main_leg['trading_up_symbol'],  # Replace with actual instrument key
        "quantity": main_leg['order_qty'],  # Quantity in lots
        "transaction_type": "SELL" if main_leg['order_action']=="S" else "BUY",
        "product": "D",  # 'D' for Delivery, 'I' for Intraday
        "price": 0  # Market price; set to 0 for market orders
    },
    {
        "instrument_key": hedge_leg['trading_up_symbol'],  # Replace with actual instrument key
        "quantity": hedge_leg['order_qty'],  # Quantity in lots
        "transaction_type": "SELL" if hedge_leg['order_action']=="S" else "BUY",
        "product": "D",  # 'D' for Delivery, 'I' for Intraday
        "price": 0  # Market price; set to 0 for market orders
    },
    ]
    margin_request = {"instruments": instruments}
    api_response = upstox_charge_api.post_margin(margin_request)
    final_margin = float(api_response.data.final_margin)
    if min_coll>final_margin*1.15:
        return orders
    else:
        margin_per_lot = 1.15*75*final_margin/float(main_leg['order_qty'])
        lots = max(0,min_coll//margin_per_lot)
        orders['Main']['order_qty']=lots*75
        orders['Hedge']['order_qty']=lots*75
        return orders


def run_hourly_trading_strategy(live,finvasia_api, upstox_opt_api, upstox_charge_api, upstox_instruments, df, entry_confirm, exit_confirm, total_profit, pos_delta, current_time=None):
    global trade_history
    put_neg_bias = 5
    entry_trade_qty= fixed_ratio_position_size(10, pos_delta, total_profit) * 75

    logging.info(f"Started STEMA Strategy")
    return_msgs=[]
    action = 'NO ACTION'
    instrument = "NSE_INDEX|Nifty 50"
    # Use current system time if not provided
    if current_time is None:
        current_time = datetime.now(ZoneInfo("Asia/Kolkata"))
    
    logging.info(f"Current time: {current_time}")
        
    result_df = calculate_supertrend(df)
    result_df=result_df.reset_index()
    
    # Get the latest row (most recent candle)
    latest_row = result_df.iloc[-1]
    latest_timestamp = latest_row['time']
    latest_close = latest_row['close']
    latest_trend = latest_row['trend']
    # latest_combined_signal = latest_row['combined_signal']
    entry_signal = latest_row['entry_signal']
    exit_signal = latest_row['exit_signal']
    rsi = latest_row['rsi']
    logging.info(f"Latest Row: {latest_row}")
    
    # Check for open orders
    open_orders = trade_history[trade_history['status'] == 'ACTIVE']
    has_open_order = not open_orders.empty
    logging.info(f"has_open_order: {has_open_order}")
    
    # # Check for trend change (compare with previous row if available)
    # if len(result_df) > 1:
    #     previous_trend = result_df.iloc[-2]['trend']
    #     exit_signal = latest_trend != previous_trend
    # else:
    #     # trend_changed = False
    #     exit_signal=False

    # logging.info(f"Previous Trend: {previous_trend}")
    logging.info(f"Current Trend: {latest_trend}")
    # logging.info(f"Trend Changed: {trend_changed}")
    logging.info(f"Entry Signal: {entry_signal}, Exit Signal:{exit_signal}")

    #Check if open order for Put exists and trend is -1 then trend changed
    logging.info(f"Checking trend change for open orders")
    if has_open_order:
        for _, order in open_orders.iterrows():
            if order['order_type'] == 'PUT' and latest_trend == -1:
                # trend_changed = True
                exit_signal=1
                logging.info(f"Trend changed for PUT order")
                break
            if order['order_type'] == 'CALL' and latest_trend == 1:
                # trend_changed = True
                exit_signal=1
                logging.info(f"Trend changed for CALL order")
                break
        
    #Check if open order for Call exists and trend is 1 then trend changed
    
    curr_pos = pd.DataFrame(finvasia_api.get_positions())
    # Exit open orders if trend changes

    if exit_signal and has_open_order:
        exit_confirm+=1
    else:
        exit_confirm=0
    
    rsi_confirm = False
    if (latest_trend ==1 and rsi>55) or (latest_trend == -1 and rsi<45):
        rsi_confirm = True


    if exit_confirm>2:# and rsi_confirm:
        exit_confirm = 0
        action = "EXIT POSITIONS"
        logging.info(f"Checking open order when trend changed")
        for _, order in open_orders.iterrows():
            order_tsm = order['trading_symbol']
            order_type = order['order_type']
            # ord_qty = min(abs(int(curr_pos[curr_pos['tsym']==order_tsm]['netqty'].iloc[0])),order['order_qty'])
            ord_qty = int(curr_pos[curr_pos['tsym']==order_tsm]['netqty'].iloc[0])
            ord_act = 'S' if order['order_action'] == 'B' else 'B'
            if (latest_trend ==-1 and order_type == 'PUT') or (latest_trend ==1 and order_type == 'CALL'):
                # Pseudocode: Close Put order
                # close_put_order(order_id, latest_close)
                # print(f"Closing Put order {order_tsm}")
                logging.info(f"Closing put order: {order}")
                # Get the current qty as per exisitng position and limit qty to that 
                ret_status, ret_msg = place_order(finvasia_api, live, order_tsm, ord_act, str(ord_qty), 'EXIT STEMA')
                return_msgs.append(ret_msg)
            
            # Update trade history
            if ret_status:
                trade_history.loc[trade_history['trading_symbol'] == order_tsm, 'status'] = 'CLOSED'
                trade_history.loc[trade_history['trading_symbol'] == order_tsm, 'exit_timestamp'] = current_time
    
    # Check for open orders again after Exit # maybe - Giving gap of 1 iteration between exit and entry
    open_orders = trade_history[trade_history['status'] == 'ACTIVE']
    has_open_order = not open_orders.empty
    logging.info(f"Check again has_open_order: {has_open_order}")

    # Place new order if no open orders and combined_signal is 1 or -1
    if not has_open_order and entry_signal != 0 and rsi_confirm:
        entry_confirm+=entry_signal
    else:
        entry_confirm=0

    if abs(entry_confirm)>2:
        entry_confirm = 0    
        orders={}
        action = 'MAKE ENTRY'
        order_type = 'CALL' if entry_signal == 1 else 'PUT'
        expiry = get_next_thursday_between_4_and_12_days(current_time)
        # Check if expiry is a holiday
        if expiry in holiday_dict:
            expiry=holiday_dict.get(expiry)
        logging.info(f"Calculated Expiry: {expiry}")
        try:
            main_leg = get_positions(upstox_opt_api, finvasia_api, instrument, expiry,entry_trade_qty,upstox_instruments, 0.35)
            logging.info(f"Main Leg: {main_leg}")
            hedge_leg = get_positions(upstox_opt_api, finvasia_api, instrument, expiry,entry_trade_qty,upstox_instruments, 0.20)
            logging.info(f"Hedge Leg: {hedge_leg}")
        except Exception as e:
            return_msgs.append({'subject': 'Error in get_positions', 'body': str(e)})
            logging.info(f"Error in get_position: {str(e)}")
            main_leg = {}
            hedge_leg = {}
            main_leg['fin_pe_symbol'] = f'{expiry}-PE-DELAT0.35'
            hedge_leg['fin_pe_symbol'] = f'{expiry}-PE-DELAT0.20'
            main_leg['fin_ce_symbol'] = f'{expiry}-CE-DELAT035'
            hedge_leg['fin_ce_symbol'] = f'{expiry}-CE-DELAT0.20'
        # Pseudocode: Place order
        # Check to not place the same trend order if exited on the same day
        # Get today's date (without time)
        today = pd.Timestamp(datetime.now(ZoneInfo("Asia/Kolkata")).date())
        # Filter rows where the date part of 'exit_timestamp' matches today
        trade_history['exit_timestamp'] = pd.to_datetime(trade_history['exit_timestamp'])
        df_today = trade_history[trade_history['exit_timestamp'].dt.date == today.date()]
        day_order_filter = list(df_today['order_type'].unique())
        # Get available cash and stock colalterals:
        limits = finvasia_api.get_limits()
        min_coll = min(float(limits['cash']) + float(limits['payin'])- float(limits['payout'])-float(limits['marginused'])/2, float(limits['collateral'])-float(limits['marginused'])/2)
        if order_type == 'PUT' and order_type not in day_order_filter:
            orders['Main']={'trading_symbol':main_leg['fin_pe_symbol'], 'trading_up_symbol':main_leg['upstox_pe_instrument_key'], 'order_action':'S', 'order_qty':str(entry_trade_qty), 'order_type':'PUT'}
            logging.info(f"Main Leg: {main_leg['fin_pe_symbol']}")
            orders['Hedge']={'trading_symbol':hedge_leg['fin_pe_symbol'], 'trading_up_symbol':hedge_leg['upstox_pe_instrument_key'], 'order_action':'B', 'order_qty':str(entry_trade_qty), 'order_type':'PUT'}
            logging.info(f"Hedge Leg: {hedge_leg['fin_pe_symbol']}")
            # Get revised trade_qty based on margin
            orders = get_revised_qty_margin(orders, upstox_charge_api, min_coll)
            # put_neg_bias
            orders['Main']['order_qty']=75*(int(orders['Main']['order_qty'])//(75*put_neg_bias))
            orders['Hedge']['order_qty']=75*(int(orders['Main']['order_qty'])//(75*put_neg_bias))
        elif order_type == 'CALL' and order_type not in day_order_filter:
            orders['Main']={'trading_symbol':main_leg['fin_ce_symbol'], 'trading_up_symbol':main_leg['upstox_ce_instrument_key'], 'order_action':'S', 'order_qty':str(entry_trade_qty), 'order_type':'CALL'}
            logging.info(f"Main Leg: {main_leg['fin_ce_symbol']}")
            orders['Hedge']={'trading_symbol':hedge_leg['fin_ce_symbol'], 'trading_up_symbol':hedge_leg['upstox_ce_instrument_key'], 'order_action':'B', 'order_qty':str(entry_trade_qty), 'order_type':'CALL'}
            logging.info(f"Hedge Leg: {hedge_leg['fin_ce_symbol']}")
            orders = get_revised_qty_margin(orders, upstox_charge_api, min_coll)
        #Place Hedge orders first
        for order_leg, order_det in orders.items():
            if order_leg == 'Hedge' and int(order_det['order_qty'])>0:
                ret_hedge_status, ret_msg=place_order(finvasia_api, live, order_det['trading_symbol'], order_det['order_action'], order_det['order_qty'], 'STEMA')
                logging.info(f"Hedge Order Status: {ret_hedge_status}")
                return_msgs.append(ret_msg)
                # Append to trade history
                new_order = pd.DataFrame({
                    'time': latest_timestamp,
                    'trading_symbol': order_det['trading_symbol'],
                    'expiry': expiry,
                    'order_action' : order_det['order_action'],
                    'order_qty': order_det['order_qty'],
                    'order_leg': order_leg,
                    'order_type': order_det['order_type'],
                    'status': 'ACTIVE',
                    'exit_timestamp': [pd.NaT]
                })
                if ret_hedge_status:
                    # trade_history = pd.concat([trade_history, new_order], ignore_index=True)
                    if not new_order.empty:
                        new_order = new_order.astype(trade_history.dtypes.to_dict(), errors='ignore')
                        logging.info(f"Update Trade History for hedge order")
                        trade_history = pd.concat([trade_history, new_order], ignore_index=True)

        #Pleace Main orders
        for order_leg, order_det in orders.items():
            if order_leg == 'Main' and int(order_det['order_qty'])>0:
                ret_main_status, ret_msg=place_order(finvasia_api, live, order_det['trading_symbol'], order_det['order_action'], order_det['order_qty'], 'STEMA')
                logging.info(f"Main Order Status: {ret_main_status}")
                return_msgs.append(ret_msg)
                # Append to trade history
                new_order = pd.DataFrame({
                    'time': latest_timestamp,
                    'trading_symbol': order_det['trading_symbol'],
                    'expiry': expiry,
                    'order_action' : order_det['order_action'],
                    'order_qty': order_det['order_qty'],
                    'order_leg': order_leg,
                    'order_type': order_det['order_type'],
                    'status': 'ACTIVE',
                    'exit_timestamp': [pd.NaT]
                })
                if ret_main_status:
                    if not new_order.empty:
                        new_order = new_order.astype(trade_history.dtypes.to_dict(), errors='ignore')
                        logging.info(f"Update Trade History for main order")
                        trade_history = pd.concat([trade_history, new_order], ignore_index=True)
    
    # Save trade history
    logging.info(f"Saving trade history")
    trade_history.to_csv(trade_history_file, index=False)
    if abs(entry_confirm) > 2 and rsi_confirm:
        action = 'Entry made'
    elif exit_confirm > 2: # and rsi_confirm:
        action = 'Closed open orders'
    else:
        action = 'No action'
    # Debug output
    subject= f"Trade Decision at {latest_timestamp}"
    email_body = f"""
    Current Time: {latest_timestamp}
    Current Close: {latest_close}
    20 EMA: {latest_row['ema20']}
    34 EMA: {latest_row['ema34']}
    Trend: {latest_trend}
    RSI: {rsi}
    Entry Signal: {entry_signal}
    Entry Confirm: {entry_confirm}
    Exit Signal: {exit_signal}
    Exit Confirm: {exit_confirm}
    Action: {action}
    """
    return_msgs.append({'subject': subject, 'body': email_body})
    # send_email_plain(subject, email_body)
    logging.info(f"sending emails: {email_body}")
    return return_msgs, entry_confirm, exit_confirm
    
def update_stema_tb(tradingsymbol, ord_type):
    global trade_history
    global trade_history_file
    otype = "PUT" if ord_type=="PE" else "CALL"
    trade_history.loc[(trade_history['trading_symbol'] == tradingsymbol) & (trade_history['order_type'] == otype), 'status'] = 'CLOSED'
    trade_history.loc[(trade_history['trading_symbol'] == tradingsymbol) & (trade_history['order_type'] == otype), 'exit_timestamp'] = datetime.now(ZoneInfo("Asia/Kolkata"))
    # Save trade history
    logging.info(f"Saving trade history")
    trade_history.to_csv(trade_history_file, index=False)

