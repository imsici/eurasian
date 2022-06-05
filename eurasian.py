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

    def __init__(self, start_day, end_day, strike, notional, average_type, putcall, cal = ql.China(ql.China.IB)):
        self.start_day = start_day
        self.end_day = end_day
        self.strike = strike
        self.calendar = cal
        self.settlement_day = self.calendar.advance(self.end_day, ql.Period("2D"))
        self.notional = notional
        self.average_type = average_type
        self.sch = self.fixing_sch()
        self.putcall = putcall

    def fixing_sch(self):
        freq = ql.Period("1D")
        conv = ql.ModifiedFollowing
        term = ql.ModifiedFollowing
        rule = ql.DateGeneration.Forward
        eom = False
        return ql.Schedule(self.start_day, self.end_day, freq, self.calendar, conv, term, rule, eom)

    def mv_mc(self, hdate, init_value, base_ts, for_ts, vol_ts, payoff, fixed_values = [], sample_size=10000):
        process = ql.BlackScholesMertonProcess(init_value, for_ts, base_ts, vol_ts)
        fixdtstmp = np.array(self.fixing_sch().dates())
        tofixdts = fixdtstmp[fixdtstmp > hdate]
        tofixivals = (tofixdts - hdate) / 365
        fixeddts = fixdtstmp[fixdtstmp <= hdate]
        assert fixeddts.size == np.array(fixed_values).size

        results = np.zeros(sample_size)

        rng = ql.UniformLowDiscrepancySequenceGenerator(process.factors() * len(self.fixing_sch().dates()))
        seqgen = ql.GaussianLowDiscrepancySequenceGenerator(rng)
        pathgen = ql.GaussianSobolPathGenerator(process, ql.TimeGrid(list(tofixivals)), seqgen, True)

        for i in range(sample_size):
            path = pathgen.next().value()
            results[i] = payoff(path) * base_ts.discount(self.settlement_day)

        return np.mean(results) * self.notional, np.std(results) * self.notional / np.sqrt(sample_size)

    def get_payoff(self):

        def european_unit_payoff_call(path):
            path = np.array(path)
            if path[-1] > self.strike:
                return path[-1] - self.strike
            else:
                return 0.

        def european_unit_payoff_put(path):
            path = np.array(path)
            if np.mean(path) < self.strike:
                return self.strike - path[-1]
            else:
                return 0.

        def eurasian_unit_payoff_call(path):
            path = np.array(path)
            if self.average_type == AverageType.ARITHMETIC:
                if np.mean(path) > self.strike:
                    return path[-1] - self.strike
                else:
                    return 0.
            elif self.average_type == AverageType.GEOMETRIC:
                if np.exp(np.log(path).mean()) > self.strike:
                    return path[-1] - self.strike
                else:
                    return 0.
            elif self.average_type == AverageType.HARMONIC:
                if np.reciprocal(np.reciprocal(path).mean()) > self.strike:
                    return path[-1] - self.strike
                else:
                    return 0.

        def eurasian_unit_payoff_put(path):
            path = np.array(path)
            if self.average_type == AverageType.ARITHMETIC:
                if np.mean(path) < self.strike:
                    return self.strike - path[-1]
                else:
                    return 0.
            elif self.average_type == AverageType.GEOMETRIC:
                if np.exp(np.log(path).mean()) < self.strike:
                    return self.strike - path[-1]
                else:
                    return 0.
            elif self.average_type == AverageType.HARMONIC:
                if np.reciprocal(np.reciprocal(path).mean()) < self.strike:
                    return self.strike - path[-1]
                else:
                    return 0.

        def regasian_unit_payoff_call(path):
            path = np.array(path)
            if self.average_type == AverageType.ARITHMETIC:
                avg_val = np.mean(path)
                if avg_val > self.strike:
                    return avg_val - self.strike
                else:
                    return 0.
            elif self.average_type == AverageType.GEOMETRIC:
                avg_val = np.exp(np.log(path).mean())
                if avg_val > self.strike:
                    return avg_val - self.strike
                else:
                    return 0.
            elif self.average_type == AverageType.HARMONIC:
                avg_val = np.reciprocal(np.reciprocal(path).mean())
                if avg_val > self.strike:
                    return avg_val - self.strike
                else:
                    return 0.

        def regasian_unit_payoff_put(path):
            path = np.array(path)
            if self.average_type == AverageType.ARITHMETIC:
                avg_val = np.mean(path)
                if avg_val < self.strike:
                    return self.strike - avg_val
                else:
                    return 0.
            elif self.average_type == AverageType.GEOMETRIC:
                avg_val = np.exp(np.log(path).mean())
                if avg_val > self.strike:
                    return self.strike - avg_val
                else:
                    return 0.
            elif self.average_type == AverageType.HARMONIC:
                avg_val = np.reciprocal(np.reciprocal(path).mean())
                if avg_val > self.strike:
                    return self.strike - avg_val
                else:
                    return 0.

        if self.putcall == PutCallType.CALL:
            return european_unit_payoff_call
        elif self.putcall == PutCallType.PUT:
            return european_unit_payoff_put
        else:
            raise TypeError("Invalid Put/Call Type")


if __name__ == '__main__':

    hdate = ql.Date(1, 6, 2022)
    init_spot = ql.QuoteHandle(ql.SimpleQuote(6.6))
    base_ts = ql.YieldTermStructureHandle(ql.FlatForward(hdate, 0.03, ql.Actual365Fixed()))
    for_ts = ql.YieldTermStructureHandle(ql.FlatForward(hdate, 0.02, ql.Actual365Fixed()))
    vol_ts = ql.BlackVolTermStructureHandle(ql.BlackConstantVol(hdate, ql.China(ql.China.IB), 0.1, ql.Actual365Fixed()))

    for k in [6.5, 6.55, 6.6, 6.65, 6.7]:
        opt = EurAsianOpt(ql.Date(2, 6, 2022), ql.Date(2, 7, 2022), k, 1., AverageType.ARITHMETIC, PutCallType.CALL)
        mv, stderr = opt.mv_mc(hdate, init_spot, base_ts, for_ts, vol_ts, opt.get_payoff())
        print("\nSPOT:         " + str(k))
        print("MARKET VALUE: " + str(mv))
        print("STD ERROR:    " + str(stderr))

