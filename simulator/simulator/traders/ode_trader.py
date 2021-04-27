from traders.trader import Trader

class ODETrader(Trader):
    def __init__(self, init_balance=0):
        self.balance = init_balance
        self.stock = 0
        self.tick_num = 0
        self.orders = []

    def respond(self, tick):
        self.tick_num += 1
        if (self.balance > tick['Ask']):
            print(self.balance)
            return ({'type': 'BID', 'price': tick['Ask'], 'quantity': tick['AskVolume']})

    def filled_order(self, order):
        self.stock += order['quantity']
        self.balance -= order['price'] * order['quantity']
        # print(self.stock, self.balance)
