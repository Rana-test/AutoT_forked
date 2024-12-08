# # 10 delta put and call across all strikes including 50 strikes
# # Adjustment starts only when the loss exceeds maximum profit
# # Adjust the IC at 40 delta till IF
# # Adjust the IF at 50 delta
# # Exit when no more adjustment possible

# # Identify session
# if trigger time between 9:00 a.m. to 12:25 p.m then its Session 1
# if trigger time between 12:30 a.m. to 3:30 p.m then its Session 2

# Session 1 : start_time: 9:15 a.m, end_time: 12:29 p.m
# Session 2 : start_time: 12:30 p.m, end_time: 15:30 p.m

# login
# while time is between start_time and end_time:
#     call monitor_trade
#     time.sleep(60)
#     Every 15mins calculate metrics (Day no, Current strikeprice of Nifty50, Current open positions, breakevens, current_profit/loss) and send mail
# logout
# update adjustment_flag in config
# if session 2:
#     update past_m2m in config by adding closed_m2m to it

# # def monitor_trade()
#     Get the current position
#     If positions exists but no. of legs is less than 2:
#         Send warning mail
#     elIf positions do not exits:
#         Send mail to make the entry manually 
#     else if no. of legs is equal to 2:
#         Identify and get the details of the PE and CE legs including ltp
#         Get the past_m2m from config
#         Calculate current profit_loss and max_profit
#         if profit_loss < -1 * max_profit:
#             Turn on the adjustmen_flag
#             Send mail alert stating entered adjustment phase
#         if adjustment_flag:
#             find the current strategy: IF or IC 
#             if IC:
#                 Calculate delta between the two legs
#                 if delta>40:
#                     Find the profit_leg and loss_leg
#                     Find the ltp of the loss_leg
#                     new_sp = Get the strikeprice of the profit_side with ltp closest to loss_leg_ltp
#                         If profit_leg is PE and new_sp is more than loss_leg strikeprice OR if profit_leg is CE and new_sp is less than loss_leg strikeprice:
#                             Caclulate adj_delta with loss_leg_ltp and profit_leg_ltp at loss_leg_stikeprice:
#                             if adj_delta> 50:
#                                 EXIT all trades
#                                 send mail stating exiting all trades
#                             else:
#                                 exit profit_leg
#                                 sell new profit_leg at loss_leg_strikeprice # strategy is now IF
#                                 send mail stating the adjustment made
#                         else:
#                             exit profit_leg
#                             sell new_profit_leg at new_sp
#                             send mail stating the adjustment made
#             elif IF:
#                 Calculate delta
#                 if delta>50:
#                     EXIT all trades
#                     send mail stating exiting all trades

import time
from datetime import datetime
import pandas as pd
from helpers import helper as h

# Configuration setup
CONFIG = {
    "adjustment_flag": 1,
    "past_m2m": 0,
    "session_1": {"start_time": "09:15", "end_time": "12:29"},
    "session_2": {"start_time": "12:30", "end_time": "23:30"},
    "nse_sym_path": "NSE_symbols.txt",
    "nfo_sym_path": "NFO_symbols.txt",
    "NiftyExpiry": "26-DEC-2024",
    "nifty_nse_token": "26000",
    "lot_size":75,
    "live": 0,
    "IC_DELTA_THRESHOLD": 20,
    "IF_DELTA_THRESHOLD": 20,
    "symbol": "NIFTY",
    'min_strike_price':22000,
    'max_strike_price':27000,
    # 'get_position_from_csv': "Positions/nifty_pos_2.csv"
}

def init():
    # Get Nifty Index Token ID
    nse_df = pd.read_csv(CONFIG.get('nse_sym_path'))
    nifty_nse_token = nse_df[(nse_df.Symbol=="Nifty 50")&(nse_df.Instrument=="INDEX")].iloc[0]['Token']
    # Get BankNifty Index Token ID
    nifty_bank_nse_token = nse_df[(nse_df.Symbol=="Nifty Bank")&(nse_df.Instrument=="INDEX")].iloc[0]['Token']
    del nse_df

    # Initialize logger
    logger = h.logger_init()
    
    # Load symbols if not present
    h.get_symbol_data()
    
    # Login
    api = h.login(logger)

    return api, logger, nifty_nse_token, nifty_bank_nse_token #, future_price_token, future_price_bnk_token

# Main execution function
def main():
    session = identify_session()
    if not session:
        print("No active trading session.")
        return
    api, logger, nifty_nse_token, nifty_bank_nse_token = init()
    start_time, end_time = CONFIG[session]["start_time"], CONFIG[session]["end_time"]
    while is_within_timeframe(start_time, end_time):
        monitor_trade(api, logger)
        time.sleep(60)
        if should_send_metrics():
            send_metrics_email()
    h.logout()
    update_adjustment_flag()
    if session == "session_2":
        update_past_m2m()

# Identify session based on current time
def identify_session():
    now = datetime.now().time()
    if is_within_timeframe("09:00", "12:25"):
        return "session_1"
    elif is_within_timeframe("12:30", "23:30"):
        return "session_2"
    return None

# Helper: Check if current time is within a timeframe
def is_within_timeframe(start, end):
    now = datetime.now().strftime("%H:%M")
    return start <= now <= end

# Send metrics email every 15 minutes
def should_send_metrics():
    return datetime.now().minute % 15 == 0

def send_metrics_email():
    # Compute and send metrics
    print("Sending metrics email...")

# Monitor trade function
def monitor_trade(api, logger):
    # Read position_data.csv
    open_positions=None
    nfo_df = pd.read_csv(CONFIG.get('nfo_sym_path'))
    if CONFIG.get("get_position_from_csv",0)!=0:
        open_positions = pd.read_csv(CONFIG.get("get_position_from_csv"))
    positions, total_m2m, closed_m2m = h.get_current_positions_new(api, logger, nfo_df, CONFIG.get('past_m2m'), open_positions)
    total_m2m += CONFIG.get('past_m2m')

    print(positions, total_m2m, closed_m2m)
    if len(positions)==0:
        send_email("No positions found. Please make manual entry.")
        return

    num_legs = len(positions)
    if num_legs < 2:
        send_email("Warning: Fewer than 2 legs in the current position.")
        return

    pe_leg, ce_leg, _ , _ = identify_legs(positions)
    max_profit = float((positions['qty'].astype(int)*positions['netupldprc'].astype(float)).sum()) * -1 + CONFIG.get('past_m2m')
    if total_m2m < -1 * max_profit:
        CONFIG["adjustment_flag"] = True
        send_email("Entered adjustment phase.")

    if CONFIG["adjustment_flag"]:
        res = api.get_quotes(exchange="NSE", token=str(CONFIG.get('nifty_nse_token')))
        current_strike = float(res['lp'])
        handle_adjustments(logger, api, nfo_df, CONFIG.get("symbol"), CONFIG.get("NiftyExpiry"), pe_leg, ce_leg, current_strike)


# Placeholder for identifying legs
def identify_legs(positions):
    pe_leg={}
    ce_leg = {}
    pe_hedge={}
    ce_hedge={}
    for i, row in positions.iterrows():
        if row['buy_sell']=='S' and row['tsym'][-6]=='P':
            pe_leg= row.to_dict()
        elif row['buy_sell']=='S' and row['tsym'][-6]=='C':
            ce_leg = row.to_dict()
        elif row['buy_sell']=='B' and row['tsym'][-6]=='C':
            ce_hedge = row.to_dict()
        elif row['buy_sell']=='B' and row['tsym'][-6]=='P':
            pe_hedge = row.to_dict()

    return pe_leg, ce_leg, pe_hedge, ce_hedge

# Handle adjustments
def handle_adjustments(logger, api, nfo_df, symbol, expiry, pe_leg, ce_leg, strike_price):
    strategy = identify_strategy(pe_leg, ce_leg, strike_price)
    if strategy == "IC":
        handle_ic_adjustment(logger, api, nfo_df, symbol, expiry, pe_leg, ce_leg)
    elif strategy == "IF":
        handle_if_adjustment(logger, api, nfo_df, symbol, expiry, pe_leg, ce_leg)

# Identify current strategy
def identify_strategy(pe_leg, ce_leg, strike_price):
    pe_ce_diff = abs(float(pe_leg['tsym'][-5:])-float(ce_leg['tsym'][-5:]))
    if (float(pe_leg['tsym'][-5:])==float(ce_leg['tsym'][-5:])) or (abs(float(pe_leg['tsym'][-5:])-strike_price)>pe_ce_diff) or (abs(float(ce_leg['tsym'][-5:])-strike_price)>pe_ce_diff):
        return "IF"
    else:
        return "IC"

# Adjust Iron Condor (IC)
def handle_ic_adjustment(logger, api, nfo_df, symbol, expiry, pe_leg, ce_leg):
    delta = calculate_delta(pe_leg, ce_leg)
    if abs(delta) > CONFIG.get("IC_DELTA_THRESHOLD"):
        # Find the profit_leg and loss_leg
        pe_leg_profit = float(pe_leg['netupldprc'])-float(pe_leg['lp'])
        ce_leg_profit = float(ce_leg['netupldprc'])-float(ce_leg['lp'])
        profit_leg=None
        loss_leg = None
        if pe_leg_profit > ce_leg_profit:
            profit_leg = pe_leg
            loss_leg = ce_leg
        else:
            profit_leg = ce_leg
            loss_leg = pe_leg
        # new_sp = Get the strikeprice of the profit_side with ltp closest to loss_leg_ltp
        # Find the ltp of the loss_leg
        optype = "PE" if profit_leg['ord_type']=="P" else "CE"
        symbolDf = pd.read_csv(CONFIG.get('nfo_sym_path'))
        profit_opt_df = h.get_Option_Chain_new(api, symbol, expiry, symbolDf, optype, min_strike_price=CONFIG.get('min_strike_price'), max_strike_price=CONFIG.get('max_strike_price'))
        new_sp_tsym, new_sp_lp = h.get_nearest_price_strike(profit_opt_df, float(loss_leg['lp']))
#     If profit_leg is PE and new_sp is more than loss_leg strikeprice OR if profit_leg is CE and new_sp is less than loss_leg strikeprice:
        if (profit_leg['ord_type']=="P" and int(new_sp_tsym[-5:]) > int(loss_leg['tsym'][-5:])) or (profit_leg['ord_type']=="C" and int(new_sp_tsym[-5:]) < int(loss_leg['tsym'][-5:])):
            handle_if_adjustment(api, symbol, expiry, pe_leg, ce_leg)
        else:
            # Think and see : if profit_leg['ord_type']=="P" and int(new_sp_tsym[-5:]) > int(loss_leg['tsym'][-5:]):

            keys_to_extract = ['buy_sell','tsym','qty','remarks', 'netupldprc']
            order_1 = {key: profit_leg[key] for key in keys_to_extract}
            order_2 = {key: profit_leg[key] for key in keys_to_extract}
            order_1['buy_sell']="B"
            order_1['qty']=abs(int(order_1['qty']))
            order_2['tsym']=new_sp_tsym
            order_df = pd.DataFrame([order_1, order_2])
            if not exit_before_adj(logger,api, nfo_df, profit_leg, loss_leg, order_1, order_2):
                h.place_order_new(logger, api, order_df, CONFIG.get("lot_size"),CONFIG.get("live"),remarks="Adjustment order")

def exit_before_adj(logger, api, nfo_df, profit_leg, loss_leg, order_1, order_2):
    adj_positions = [profit_leg, loss_leg, order_1, order_2]
    max_profit = CONFIG.get('past_m2m')
    for pos in adj_positions:
        max_profit += float(pos['netupldprc'])*int(pos['qty'])*-1
    if max_profit <0:
        exit_all_trades(logger,api, nfo_df)
        return True
    else:
        return False

# Adjust Iron Fly (IF)
def handle_if_adjustment(logger, api, nfo_df, symbol, expiry, pe_leg, ce_leg):
    delta = calculate_delta(pe_leg, ce_leg)
    if delta > CONFIG.get("IF_DELTA_THRESHOLD"):
        # Find the profit_leg and loss_leg
        pe_leg_profit = float(pe_leg['netupldprc'])-float(pe_leg['lp'])
        ce_leg_profit = float(ce_leg['netupldprc'])-float(ce_leg['lp'])
        profit_leg=None
        loss_leg = None
        if pe_leg_profit > ce_leg_profit:
            profit_leg = pe_leg
            loss_leg = ce_leg
        else:
            profit_leg = ce_leg
            loss_leg = pe_leg
        # new_sp = Get the strikeprice of the profit_side with ltp closest to loss_leg_ltp
        # Find the ltp of the profit_leg
        optype = "PE" if loss_leg['ord_type']=="P" else "CE"
        symbolDf = pd.read_csv(CONFIG.get('nfo_sym_path'))
        loss_opt_df = h.get_Option_Chain_new(api, symbol, expiry, symbolDf, optype, min_strike_price=CONFIG.get('min_strike_price'), max_strike_price=CONFIG.get('max_strike_price'))
        new_sp_tsym, new_sp_lp = h.get_nearest_price_strike(loss_opt_df, float(profit_leg['lp']))
        keys_to_extract = ['buy_sell','tsym','qty','remarks', 'netupldprc']
        order_1 = {key: loss_leg[key] for key in keys_to_extract}
        order_2 = {key: loss_leg[key]for key in keys_to_extract}
        order_1['buy_sell']="B"
        order_1['qty']=abs(int(order_1['qty']))
        order_2['tsym']=new_sp_tsym
        order_df = pd.DataFrame([order_1, order_2])
        h.place_order_new(logger, api, order_df, CONFIG.get("lot_size"),CONFIG.get("live"),remarks="Adjustment order")

        # exit_all_trades("Delta > 50; exiting all trades.")

# Calculate delta between legs
def calculate_delta(pe_leg, ce_leg):
    delta = 100*(float(pe_leg['netupldprc'])-float(ce_leg['netupldprc']))/(float(pe_leg['netupldprc'])+float(ce_leg['netupldprc']))
    return delta

# Exit all trades
def exit_all_trades(logger, api, nfo_df):
    if CONFIG.get("get_position_from_csv")!=0:
        open_positions = pd.read_csv(CONFIG.get("get_position_from_csv"))
    positions, total_m2m, closed_m2m = h.get_current_positions_new(api, logger, nfo_df, CONFIG.get('past_m2m'), open_positions)
    positions['buy_sell']=positions['buy_sell'].apply(lambda x: 'B' if x == 'S' else 'S')
    positions['qty']=positions['qty'].apply(lambda x: abs(x))
    h.place_order_new(logger, api, positions, CONFIG.get("lot_size"),CONFIG.get("live"),remarks="Exit order")

if __name__ == "__main__":
    main()
