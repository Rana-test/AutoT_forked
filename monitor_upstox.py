from helpers import helper as h
# Check the current position
# Get metrics
# mtm%
# %age distance from nearest breakeven
# Delta
# strike price
# breakevens
# days to expiry

#Adjustments
# if adjustment required
# send mail every 1 min

def main():
# Start at 9:00 and end at 3:30
    session = h.identify_session()
    if not session:
        print("No active trading session.")
        return
    
    email_subject ="Upstox: "
# Send mail every 10 mins

if __name__ =="__main__":
    main()