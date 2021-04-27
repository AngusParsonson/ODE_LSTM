from time import sleep
from utils.data_utils import TradingDataLoader
from traders.ode_trader import ODETrader

if __name__ == '__main__':
    data_loader = TradingDataLoader(r"C:\Users\Angus Parsonson\Documents\University\Fourth Year\Dissertation\data\WTB\WTB.GBGBX_Ticks_31.03.2021-31.03.2021.csv")
    
    traders = []
    orders = []
    traders.append(ODETrader(init_balance=4000))
    curr_tick = data_loader.step()
    speed = 1
    while(curr_tick[1]):
        # Main loop
        for i, t in enumerate(traders):
            order = t.respond(curr_tick[0])
            if (order == None): continue
            orders.append((i, order))
            
        unfilled = []
        while (len(orders) > 0):
            order = orders.pop(0)
            if (order[1]['type'] == 'BID'):
                if (order[1]['price'] >= curr_tick[0]['Ask'] and order[1]['quantity'] <= curr_tick[0]['AskVolume']):
                    traders[order[0]].filled_order(order[1])
                else:
                    unfilled.append(order)
            else:
                print("received ask from trader: " + str(order[0]))
        orders = unfilled
        print(orders)
        # Next tick
        sleep(curr_tick[1]/speed)
        curr_tick = data_loader.step()
        