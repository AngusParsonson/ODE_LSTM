from abc import ABC, abstractmethod

class Trader(ABC):
    @abstractmethod
    def respond(self, tick):
        '''Called by the main loop at every timestep

        Args:
            tick: current top of limit order book (level 1)
                tick['Local time'] = time in seconds since start of trading day
                tick['Ask'] = current best ask price
                tick['Bid'] = current best bid price
                tick['AskVolume'] = current best ask volume
                tick['BidVolume'] = current best bid volume
                tick['DeltaT'] = time in seconds since last tick
        Returns:
            A dict representing a message to the order book 
            {
                type: "BID" or "ASK",
                price: float,
                quantity: float
            }
        '''
        pass
    
    @abstractmethod
    def filled_order(self, order):
        '''Called by the main loop whenever a trader's order is filled

        Args:
            order: A dict representing the fulfilled order 
            {
                type: "BID" or "ASK",
                price: float,
                quantity: float
            }
        '''
        pass
