# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import numpy as np
import QuantLib as ql
from enum import Enum


class AverageType(Enum):
    ARITHMETIC = 1
    GEOMETRIC = 2
    HARMONIC = 3

class PutCallType(Enum):
    CALL = 1
    PUT = 2


class EurAsianOpt:

    def __init__(self, start_day, end_day, strike, notional, average_type):
        self.start_day = start_day
        self.end_day = end_day
        self.strike = strike
        self.calendar = ql.China(ql.China.IB)
        self.settlement_day = self.calendar.advance(self.end_day, ql.Period("2D"))
        self.notional = notional
        self.average_type = average_type
        self.sch = self.fixing_sch()

    def fixing_sch(self):
        cal = ql.China(ql.China.IB)
        freq = ql.Period("1D")
        conv = ql.ModifiedFollowing
        term = ql.ModifiedFollowing
        rule = ql.DateGeneration.Forward
        eom = False
        return ql.Schedule(self.start_day, self.end_day, freq, cal, conv, term, rule, eom)

    def npv(self, type='MC'):
        if type == 'MC':
            return npv_mc()
        else:
            return npv_pde()

    def npv_mc(self, hdate, init_value, base_ts, for_ts, vol_ts, payoff, fixed_values, sample_size=20000):
        process = ql.BlackScholesMertonProcess(init_value, for_ts, base_ts, vol_ts)
        fixdtstmp = np.array(self.fixing_sch().dates())
        tofixdts = fixdtstmp[fixdtstmp > hdate]
        tofixivals = (tofixdts - hdate) / 365
        fixeddts = fixdtstmp[fixdtstmp <= hdate]
        assert fixeddts.size == np.array(fixed_values).size

        results = np.zeros(sample_size)

        rng = ql.UniformLowDiscrepancySequenceGenerator(process.factors() * (len(self.fixing_sch().dates()) - 1))
        seqgen = ql.GaussianLowDiscrepancySequenceGenerator(rng)
        pathgen = ql.GaussianSobolPathGenerator(process, list(fixivals), seqgen, True)

        for i in range(sample_size):
            path = pathgen.next().value()
            results[i] = payoff(path)

        return np.mean(results), np.std(results)

    def npv_pde(self):
        pass


def eurasian_unit_payoff_call(path, strike):
    path = np.array(path)
    if np.mean(path) > strike:
        return path[-1] - strike
    else:
        return 0.

def eurasian_unit_payoff_put(path, strike):
    path = np.array(path)
    if np.mean(path) < strike:
        return strike - path[-1]
    else:
        return 0.

def stdasian_unit_payoff_call(path, strike):
    path = np.array(path)
    if np.mean(path) > strike:
        return np.mean(path) - strike
    else:
        return 0.

def stdasian_unit_payoff_call(path, strike):
    path = np.array(path)
    if np.mean(path) > strike:
        return np.mean(path) - strike
    else:
        return 0.

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    opt = EurAsianOpt(ql.Date(2, 6, 2022), ql.Date(2, 7, 2022), 6.6, 10000000., AverageType.ARITHMETIC)
    hdate = ql.Date(1, 6, 2022)


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
