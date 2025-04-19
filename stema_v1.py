import pandas as pd
import numpy as np
from api_helper import get_time
from datetime import datetime, timedelta, time
import math
import calendar
import os
from datetime import datetime, timedelta
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

def round_to_previous_15_45(dt):
    hour = dt.hour
    minute = dt.minute

    # Only apply rounding if time is between 9:15 and 15:30
    start = dt.replace(hour=9, minute=15, second=0, microsecond=0)
    end = dt.replace(hour=15, minute=30, second=0, microsecond=0)

    if dt < start:
        return start
    if dt > end:
        return end

    # Round down to previous 15 or 45 minute
    if minute >= 45:
        rounded_minute = 45
    elif minute >= 15:
        rounded_minute = 15
    else:
        # If before 15 past, round down to the previous hour at 45
        dt -= timedelta(hours=1)
        rounded_minute = 45

    return dt.replace(minute=rounded_minute, second=0, microsecond=0)

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
    strike_interval = 50
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

def get_data(api, now):
    """
    Fetch 30 days of hourly candle data for Nifty 50 from Finvasia API and save to test.csv.
    
    Parameters:
    api: Finvasia API client instance
    now (datetime): Current timestamp
    
    Returns:
    pd.DataFrame: DataFrame with columns ['time', 'open', 'high', 'low', 'close']
    """
    nifty_token = '26000'  # NSE|26000 is the Nifty 50 index
    
    # Round current time (assuming round_to_previous_15_45 is defined elsewhere)
    if now is None:
        now = datetime.now()
        
    rounded_now = now  # Note: Original code suggests rounding function, adjust if needed

    # Time 30 days ago
    rounded_thirty_days_ago = rounded_now - timedelta(days=21)

    # Desired format
    fmt = "%d-%m-%Y %H:%M:%S"

    # Convert times to seconds (assuming get_time is defined elsewhere)
    start_secs = get_time(rounded_thirty_days_ago.strftime(fmt))  # dd-mm-YYYY HH:MM:SS
    end_secs = get_time(rounded_now.strftime(fmt))

    # Fetch data from Finvasia API
    bars = api.get_time_price_series(
        exchange='NSE',
        token=nifty_token,
        starttime=int(start_secs),
        endtime=int(end_secs),
        interval=60  # 60-minute candles
    )

    # Create DataFrame
    df = pd.DataFrame.from_dict(bars)
    df.rename(columns={
        'into': 'open',
        'inth': 'high',
        'intl': 'low',
        'intc': 'close'
    }, inplace=True)
    
    # Convert time to datetime and format
    df['time'] = pd.to_datetime(df['time'], dayfirst=True)
    df['time'] = df['time'].dt.strftime('%d-%m-%Y %H:%M:%S')
    
    # Select and convert columns to float
    df = df[['time', 'open', 'high', 'low', 'close']]
    df[['open', 'high', 'low', 'close']] = df[['open', 'high', 'low', 'close']].astype(float)
    
    # Sort by time in ascending order
    df['time_dt'] = pd.to_datetime(df['time'], format='%d-%m-%Y %H:%M:%S')
    df = df.sort_values('time_dt').reset_index(drop=True)
    df = df.drop(columns=['time_dt'])
    
    # Save to test.csv
    df.to_csv('test.csv', index=False)
    
    return df

def place_order(api, live, trading_symbol, buy_sell, qty, order_type):
    quantity = qty
    tradingsymbol= trading_symbol
    prd_type = 'M'
    exchange = 'NFO' 
    # disclosed_qty= 0
    price_type = 'MKT'
    price=0
    trigger_price = None
    retention='DAY'
    subject = 'Order Placement'
    body = ''
    if live:
        response = api.place_order(buy_or_sell=buy_sell, product_type=prd_type, exchange=exchange, tradingsymbol=tradingsymbol, quantity=quantity, discloseqty=quantity,price_type=price_type, price=price,trigger_price=trigger_price, retention=retention, remarks=order_type)
        if response is None or 'norenordno' not in response:
            return False, {'subject': "Order Placement Failed", 'body': "Order Placement Failed"}
        order_id = response['norenordno']
        email_body = f"Order placed successfully : Order No: {order_id}/n"
        for _ in range(10):  # Try for ~10 seconds
            time.sleep(1)
            orders = api.get_order_book()
            if orders:
                matching_orders = [o for o in orders if o['norenordno'] == order_id]
                if matching_orders:
                    order = matching_orders[0]
                    status = order['status']
                    if status == 'COMPLETE':
                        # subject = f"Order executed successfully.: Order No: {order_id}"
                        body+=f"Order executed successfully.: Order No: {order_id}"
                        return True, {'subject': subject, 'body': email_body}
                    elif status in ['REJECTED', 'CANCELLED']:
                        print(f"Order {status}. Reason: {order.get('rejreason', 'Not available')}")
                        return False, {'subject': subject, 'body': email_body}
            else:
                email_body = email_body+ f"Could not fetch order book./n"

        email_body = email_body+ "Timed out waiting for order update."
        return True, {'subject': subject, 'body': email_body}

    else:
        print(f'buy_or_sell={buy_sell}, product_type={prd_type}, exchange={exchange}, tradingsymbol={tradingsymbol}, quantity={quantity}, discloseqty={quantity},price_type={price_type}, price={price},trigger_price={trigger_price}, retention={retention}, remarks={order_type}')
        subject = f"{order_type} order for {tradingsymbol}"
        email_body = f"{order_type} order for {tradingsymbol}"
        # send_email_plain(subject, email_body)
        return True, {'subject': subject, 'body': email_body}

def calculate_supertrend_and_ema(df, atr_period=10, multiplier=3.5, ema_period=130):
    """
    Calculate Supertrend indicator, 20-day EMA, and combined signals for a given OHLC dataframe.
    
    Parameters:
    - df (pd.DataFrame): DataFrame with columns 'time', 'open', 'high', 'low', 'close'
    - atr_period (int): Period for ATR calculation, default is 10
    - multiplier (float): Multiplication factor for bands, default is 3.5
    - ema_period (int): Period for EMA calculation, default is 130 (approx. 20 trading days)
    
    Returns:
    - pd.DataFrame: DataFrame with added 'supertrend', 'trend', 'signal', 'ema', and 'combined_signal' columns
    """
    # Ensure dataframe is sorted by timestamp
    df = df.sort_values('time').reset_index(drop=True)
    
    # Calculate True Range (TR)
    df['tr'] = np.maximum(df['high'] - df['low'],
                          np.maximum(abs(df['high'] - df['close'].shift(1)),
                                     abs(df['low'] - df['close'].shift(1))))
    df.loc[0, 'tr'] = df.loc[0, 'high'] - df.loc[0, 'low']  # First row TR
    
    # Calculate ATR using EMA (matches TradingView's RMA)
    alpha = 1 / atr_period
    df['atr'] = df['tr'].ewm(alpha=alpha, adjust=False).mean()
    
    # Calculate (High + Low)/2
    df['hl2'] = (df['high'] + df['low']) / 2
    
    # Calculate basic upper and lower bands
    df['basic_upper_band'] = df['hl2'] + multiplier * df['atr']
    df['basic_lower_band'] = df['hl2'] - multiplier * df['atr']
    
    # Calculate 20-day EMA (130 periods for hourly data)
    df['ema'] = df['close'].ewm(span=ema_period, adjust=False).mean()
    
    # Initialize arrays
    supertrend = np.zeros(len(df))
    trend = np.zeros(len(df))
    final_upper_band = np.zeros(len(df))
    final_lower_band = np.zeros(len(df))
    
    # Set initial values
    final_upper_band[0] = df['basic_upper_band'].iloc[0]
    final_lower_band[0] = df['basic_lower_band'].iloc[0]
    trend[0] = 1 if df['close'].iloc[0] <= df['basic_upper_band'].iloc[0] else -1
    supertrend[0] = final_lower_band[0] if trend[0] == 1 else final_upper_band[0]
    
    # Iterate through remaining rows
    for i in range(1, len(df)):
        # Adjust final bands based on previous trend and close
        if trend[i-1] == 1:
            final_lower_band[i] = max(df['basic_lower_band'].iloc[i], final_lower_band[i-1]) if df['close'].iloc[i-1] > final_lower_band[i-1] else df['basic_lower_band'].iloc[i]
            final_upper_band[i] = df['basic_upper_band'].iloc[i]
        else:
            final_upper_band[i] = min(df['basic_upper_band'].iloc[i], final_upper_band[i-1]) if df['close'].iloc[i-1] < final_upper_band[i-1] else df['basic_upper_band'].iloc[i]
            final_lower_band[i] = df['basic_lower_band'].iloc[i]
        
        # Determine trend direction based on close vs. previous Supertrend
        if trend[i-1] == 1 and df['close'].iloc[i] < supertrend[i-1]:
            trend[i] = -1
        elif trend[i-1] == -1 and df['close'].iloc[i] > supertrend[i-1]:
            trend[i] = 1
        else:
            trend[i] = trend[i-1]
        
        # Set Supertrend value
        supertrend[i] = final_lower_band[i] if trend[i] == 1 else final_upper_band[i]
    
    # Add Supertrend and trend to DataFrame
    df['supertrend'] = supertrend.round(0).astype(int)
    df['trend'] = trend.astype(int)
    
    # Generate trend flip signals
    trend_array = df['trend'].to_numpy()
    signals = np.zeros(len(df), dtype=int)
    signals[1:] = np.where(trend_array[1:] > trend_array[:-1], 1,
                           np.where(trend_array[1:] < trend_array[:-1], -1, 0))
    df['signal'] = signals
    
    # Generate combined signals based on Close, EMA, and Supertrend
    combined_signals = np.zeros(len(df), dtype=int)
    close = df['close'].to_numpy()
    ema = df['ema'].to_numpy()
    supertrend = df['supertrend'].to_numpy()
    combined_signals = np.where((close > ema) & (close > supertrend), 1,
                               np.where((close < ema) & (close < supertrend), -1, 0))
    df['combined_signal'] = combined_signals
    df.to_csv('temp_data.csv', index=False)
    # Clean up intermediate columns
    df = df.drop(columns=['tr', 'atr', 'hl2', 'basic_upper_band', 'basic_lower_band'])
    
    return df

def run_hourly_trading_strategy(live, trade_qty, finvasia_api, upstox_opt_api, upstox_instruments, df, trade_history_file='trade_history_stema.csv', current_time=None):
    # live=False
    # trade_qty=75
    # upstox_instruments = pd.read_csv("https://assets.upstox.com/market-quote/instruments/exchange/complete.csv.gz")
    return_msgs=[]
    instrument = "NSE_INDEX|Nifty 50"
    # Use current system time if not provided
    if current_time is None:
        current_time = datetime.now()
    
    # # Define trading hours (9:16 a.m. to 3:16 p.m.)
    # start_time = time(9, 16)
    # end_time = time(17, 20)
    # current_hour = current_time.time()
    
    # # Check if within trading hours
    # if not (start_time <= current_hour <= end_time):
    #     print(f"Outside trading hours: {current_time}")
    #     return
    
    # Ensure timestamp is in datetime format
    if not pd.api.types.is_datetime64_any_dtype(df['time']):
        # df['time'] = pd.to_datetime(df['time'])
        df['time'] = pd.to_datetime(df['time'], format='%d-%m-%Y %H:%M:%S', errors='coerce')
        # '%d-%m-%Y %H:%M:%S'
    
    # Filter OHLC data up to current time
    df = df[df['time'] <= current_time].copy()
    if df.empty:
        print("No data available up to current time")
        return
    
    # Calculate Supertrend, EMA, and signals
    result_df = calculate_supertrend_and_ema(df)
    
    # Get the latest row (most recent candle)
    latest_row = result_df.iloc[-1]
    latest_timestamp = latest_row['time']
    latest_close = latest_row['close']
    latest_trend = latest_row['trend']
    latest_combined_signal = latest_row['combined_signal']
    
       # Read trade history
    if os.path.exists(trade_history_file):
        trade_history = pd.read_csv(trade_history_file, parse_dates=['time', 'exit_timestamp'])
    else:
        trade_history = pd.DataFrame(columns=['time', 'trading_symbol', 'order_type', 'order_action', 'order_leg', 'status','exit_timestamp'])
    
    # Check for open orders
    open_orders = trade_history[trade_history['status'] == 'ACTIVE']
    has_open_order = not open_orders.empty
    
    # Check for trend change (compare with previous row if available)
    if len(result_df) > 1:
        previous_trend = result_df.iloc[-2]['trend']
        trend_changed = latest_trend != previous_trend
    else:
        trend_changed = False

    #Check if open order for Put exists and trend is -1 then trend changed
    if has_open_order:
        for _, order in open_orders.iterrows():
            if order['order_type'] == 'PUT' and latest_trend == -1:
                trend_changed = True
                break
            if order['order_type'] == 'CALL' and latest_trend == 1:
                trend_changed = True
                break
            if order['order_type'] == 'PUT' and previous_trend == -1:
                trend_changed = True
                break
            if order['order_type'] == 'CALL' and previous_trend == 1:
                trend_changed = True
                break
        
    #Check if open order for Call exists and trend is 1 then trend changed
    
    # Exit open orders if trend changes
    if trend_changed and has_open_order:
        for _, order in open_orders.iterrows():
            order_tsm = order['trading_symbol']
            order_type = order['order_type']
            ord_act = 'S' if order['order_action'] == 'B' else 'B'
            if latest_trend ==-1 and order_type == 'PUT':
                # Pseudocode: Close Put order
                # close_put_order(order_id, latest_close)
                print(f"Closing Put order {order_tsm}")
                ret_status, ret_msg = place_order(finvasia_api, live, order['trading_symbol'], ord_act, str(order['order_qty']), 'EXIT STEMA')
                return_msgs.append(ret_msg)
            elif latest_trend ==1 and order_type == 'CALL':
                # Pseudocode: Close Call order
                # close_call_order(order_id, latest_close)
                print(f"Closing Call order {order_tsm}")
                ret_status, ret_msg = place_order(finvasia_api, live, order['trading_symbol'], ord_act, str(order['order_qty']), 'EXIT STEMA')
                return_msgs.append(ret_msg)
            
            # Update trade history
            if ret_status:
                trade_history.loc[trade_history['trading_symbol'] == order_tsm, 'status'] = 'CLOSED'
                trade_history.loc[trade_history['trading_symbol'] == order_tsm, 'exit_timestamp'] = current_time
    
    # Place new order if no open orders and combined_signal is 1 or -1
    if not has_open_order and latest_combined_signal != 0:
        orders={}
        order_type = 'PUT' if latest_combined_signal == 1 else 'CALL'
        expiry = get_next_thursday_between_4_and_12_days(current_time)
        # Check if expiry is a holiday
        if expiry in holiday_dict:
            expiry=holiday_dict.get(expiry)
        try:
            main_leg = get_positions(upstox_opt_api, finvasia_api, instrument, expiry,trade_qty,upstox_instruments, 0.4)
            hedge_leg = get_positions(upstox_opt_api, finvasia_api, instrument, expiry,trade_qty,upstox_instruments, 0.25)
        except Exception as e:
            return_msgs.append({'subject': 'Error in get_positions', 'body': str(e)})
            main_leg = {}
            hedge_leg = {}
            main_leg['fin_pe_symbol'] = f'{expiry}-PE-DELAT0.4'
            hedge_leg['fin_pe_symbol'] = f'{expiry}-PE-DELAT0.25'
            main_leg['fin_ce_symbol'] = f'{expiry}-CE-DELAT0.4'
            hedge_leg['fin_ce_symbol'] = f'{expiry}-CE-DELAT0.25'
        # Pseudocode: Place order
        if order_type == 'PUT':
            orders['Main']={'trading_symbol':main_leg['fin_pe_symbol'], 'order_action':'S', 'order_qty':str(trade_qty), 'order_type':'PUT'}
            orders['Hedge']={'trading_symbol':hedge_leg['fin_pe_symbol'], 'order_action':'B', 'order_qty':str(trade_qty), 'order_type':'PUT'}
        else:
            orders['Main']={'trading_symbol':main_leg['fin_ce_symbol'], 'order_action':'S', 'order_qty':str(trade_qty), 'order_type':'CALL'}
            orders['Hedge']={'trading_symbol':hedge_leg['fin_ce_symbol'], 'order_action':'B', 'order_qty':str(trade_qty), 'order_type':'CALL'}
        
        #Place Hedge orders first
        for order_leg, order_det in orders.items():
            if order_leg == 'Hedge':
                ret_hedge_status, ret_msg=place_order(finvasia_api, live, order_det['trading_symbol'], order_det['order_action'], order_det['order_qty'], 'STEMA')
                return_msgs.append(ret_msg)
                # Append to trade history
                new_order = pd.DataFrame({
                    'time': latest_timestamp,
                    'trading_symbol': order_det['trading_symbol'],
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
                        trade_history = pd.concat([trade_history, new_order], ignore_index=True)

        #Pleace Main orders
        for order_leg, order_det in orders.items():
            if order_leg == 'Main':
                ret_main_status, ret_msg=place_order(finvasia_api, live, order_det['trading_symbol'], order_det['order_action'], order_det['order_qty'], 'STEMA')
                return_msgs.append(ret_msg)
                # Append to trade history
                new_order = pd.DataFrame({
                    'time': latest_timestamp,
                    'trading_symbol': order_det['trading_symbol'],
                    'order_action' : order_det['order_action'],
                    'order_qty': order_det['order_qty'],
                    'order_leg': order_leg,
                    'order_type': order_det['order_type'],
                    'status': 'ACTIVE',
                    'exit_timestamp': [pd.NaT]
                })
                if ret_main_status:
                    trade_history = pd.concat([trade_history, new_order], ignore_index=True)
    
    # Save trade history
    trade_history.to_csv(trade_history_file, index=False)
    
    # Debug output
    subject= f"Trade Decision at {latest_timestamp}"
    email_body = f"""
    Current Time: {latest_timestamp}
    Current Close: {latest_close}
    Supertrend: {latest_row['supertrend']}
    EMA: {latest_row['ema']}
    Trend: {latest_trend}
    Combined Signal: {latest_combined_signal}
    Action: {'Placed ' + order_type if not has_open_order and latest_combined_signal != 0 else 'No action' if not trend_changed else 'Closed open orders'}
    """
    return_msgs.append({'subject': subject, 'body': email_body})
    # send_email_plain(subject, email_body)
    return return_msgs
    # print(f"Current Time: {latest_timestamp}")
    # print(f"Close: {latest_close}")
    # print(f"Supertrend: {latest_row['supertrend']}")
    # print(f"EMA: {latest_row['ema']}")
    # print(f"Trend: {latest_trend}")
    # print(f"Combined Signal: {latest_combined_signal}")
    # print(f"Action: {'Placed ' + order_type if not has_open_order and latest_combined_signal != 0 else 'No action' if not trend_changed else 'Closed open orders'}")

