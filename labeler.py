import numpy as np
import pandas as pd

class AmplitudeBasedLabeler:
    def __init__(self, minamp, Tinactive):
        """
        minamp: minimum move amplitude in basis points (e.g. 15 = 15bps)
        Tinactive: number of bars of inactivity to signal end of move
        """
        self.minamp = minamp / 10000.0  # Convert bps to decimal
        self.Tinactive = Tinactive
        self.df = None

    def _find_extrema(self):
        """
        Finds price swing points based on min amplitude and inactivity.
        This is the core logic of the labeler.
        """
        df = self.df
        T, P = df.index, df.price.values
        n = len(P)
        
        #
        # find swings based on min return
        #
        extrema = np.zeros(n)
        extrema_p = np.zeros(n)
        extrema_t = np.zeros(n, dtype=np.int64)

        # first pass, find major swings
        last_extrema_p = P[0]
        last_extrema_t = 0
        up = True
        down = True
        k = 0
        for i in range(1, n):
            ret = P[i] / last_extrema_p - 1.0
            if up and ret > self.minamp:
                # new high, record previous low
                extrema[k] = -1
                extrema_p[k] = last_extrema_p
                extrema_t[k] = last_extrema_t
                k += 1
                last_extrema_p = P[i]
                last_extrema_t = i
                up = True
                down = False
            elif down and ret < -self.minamp:
                # new low, record previous high
                extrema[k] = 1
                extrema_p[k] = last_extrema_p
                extrema_t[k] = last_extrema_t
                k += 1
                last_extrema_p = P[i]
                last_extrema_t = i
                up = False
                down = True
            else:
                if up and P[i] < last_extrema_p:
                    last_extrema_p = P[i]
                    last_extrema_t = i
                elif down and P[i] > last_extrema_p:
                    last_extrema_p = P[i]
                    last_extrema_t = i
        
        #
        # find swings based on max inactivity
        #
        extrema2 = np.zeros(k)
        extrema_p2 = np.zeros(k)
        extrema_t2 = np.zeros(k, dtype=np.int64)

        last_extrema = extrema[0]
        last_extrema_p = extrema_p[0]
        last_extrema_t = extrema_t[0]

        last_high = -1
        last_high_t = -1
        last_low = -1
        last_low_t = -1
        m = 0
        for i in range(1, k):
            if extrema[i] == -1:
                # found a low
                if last_extrema == -1:
                    if last_high_t == -1 or extrema_p[i - 1] > last_high:
                        last_high = extrema_p[i - 1]
                        last_high_t = extrema_t[i - 1]
                else: # last_extrema == 1
                    if i - last_low_t > self.Tinactive:
                        extrema2[m] = -1
                        extrema_p2[m] = last_low
                        extrema_t2[m] = last_low_t
                        m += 1

                    last_high = extrema_p[i - 1]
                    last_high_t = extrema_t[i - 1]

                last_extrema = -1
                last_low = extrema_p[i]
                last_low_t = extrema_t[i]
            else: # extrema[i] == 1
                # found a high
                if last_extrema == 1:
                    if last_low_t == -1 or extrema_p[i - 1] < last_low:
                        last_low = extrema_p[i - 1]
                        last_low_t = extrema_t[i - 1]
                else: # last_extrema == -1
                    if i - last_high_t > self.Tinactive:
                        extrema2[m] = 1
                        extrema_p2[m] = last_high
                        extrema_t2[m] = last_high_t
                        m += 1

                    last_low = extrema_p[i - 1]
                    last_low_t = extrema_t[i - 1]

                last_extrema = 1
                last_high = extrema_p[i]
                last_high_t = extrema_t[i]

        return extrema_t2[:m], extrema_p2[:m], extrema2[:m]

    def transform(self, prices):
        """
        Labels the price series based on the detected swing points.
        :param prices: A numpy array of prices.
        :return: A numpy array of labels (+1 for up, -1 for down, 0 for neutral).
        """
        self.df = pd.DataFrame({'price': prices})
        extrema_t, extrema_p, extrema = self._find_extrema()
        
        labels = np.zeros(len(prices))
        for i in range(len(extrema_t) - 1):
            start_idx = extrema_t[i]
            end_idx = extrema_t[i+1]
            # If the extremum is a high (1), the trend leading to it was up.
            # If the extremum is a low (-1), the trend leading to it was down.
            if extrema[i+1] == 1:
                labels[start_idx:end_idx] = 1
            elif extrema[i+1] == -1:
                labels[start_idx:end_idx] = -1
        
        return labels
