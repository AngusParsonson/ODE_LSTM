from time import sleep
from utils.data_utils import TradingDataLoader
from traders.ode_trader import ODETrader

if __name__ == '__main__':
    data_loader = TradingDataLoader(r"C:\Users\Angus Parsonson\Documents\University\Fourth Year\Dissertation\data\WTB\WTB.GBGBX_Ticks_30.03.2021-30.03.2021.csv")
    traders = []
    orders = []
    traders.append(ODETrader(init_balance=4000))
    curr_tick = data_loader.step()
    speed = float('inf')
    while(curr_tick[1]):
        market_price = ((curr_tick[0]['Bid'] * curr_tick[0]['AskVolume'] + 
                    curr_tick[0]['Ask'] * curr_tick[0]['BidVolume']) / 
                    (curr_tick[0]['AskVolume'] + curr_tick[0]['BidVolume']))
        # print(orders)
        # Main loop
        for i, t in enumerate(traders):
            t_orders = t.respond(curr_tick[0])
            if (t_orders == None): continue
            else:
                for o in t_orders:
                    orders.append((i, o))
            
        unfilled = []
        while (len(orders) > 0):
            order = orders.pop(0)
            if (order[1]['type'] == 'BID'):
                if (order[1]['price'] >= market_price and 
                    order[1]['quantity'] <= curr_tick[0]['AskVolume']):
                    order[1]['price'] = market_price
                    traders[order[0]].filled_order(order[1])
                else:
                    unfilled.append(order)
            else:
                if (order[1]['price'] <= market_price and 
                    order[1]['quantity'] <= curr_tick[0]['BidVolume']):
                    order[1]['price'] = market_price
                    traders[order[0]].filled_order(order[1])
                else:
                    unfilled.append(order)
                # print("received ask from trader: " + str(order[0]))
        orders = unfilled
        # print(orders)
        # Next tick
        if (speed != float('inf')):
            sleep(curr_tick[1]/speed)
        curr_tick = data_loader.step()
    for t in traders:
        t.print_vals()
        