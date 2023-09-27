

import math
import random

#LAPLACE
class LaplaceDistribution:
    def __init__(self, rand, loc, scale):
        self.rand = rand
        self.loc = loc
        self.scale = scale

    def pdf(self, x):
        if self.scale <= 0:
            raise ValueError("Skála értéke pozitívnak kell lennie.")

        abs_diff = abs(x - self.loc)
        probability = (1 / (2 * self.scale)) * math.exp(-abs_diff / self.scale)
        return probability

    def cdf(self, x):
        if self.scale <= 0:
            raise ValueError("Skála értéke pozitívnak kell lennie.")

        if x < self.loc:
            probability = 0.5 * math.exp((x - self.loc) / self.scale)
        else:
            probability = 1 - 0.5 * math.exp(-(x - self.loc) / self.scale)
        return probability

    def ppf(self, p):
        if p < 0 or p > 1:
            raise ValueError("Az 'p' értéke 0 és 1 között kell legyen.")

        return self.loc - self.scale * math.copysign(1, p - 0.5) * math.log(1 - 2 * abs(p - 0.5))

    def gen_random(self):
        u1 = self.rand.random()
        u2 = self.rand.random()

        if u2 <= 0.5:
            X = self.loc - self.scale * math.log(2 * u1)
        else:
            X = self.loc + self.scale * math.log(2 * (1 - u1))

        return X

    def mean(self):
        if self.scale == 0:
            raise Exception("Moment undefined")
        return self.loc

    def variance(self):
        if self.scale == 0:
            raise Exception("Moment undefined")
        return 2 * self.scale ** 2

    def skewness(self):
        if self.scale == 0:
            raise Exception("Moment undefined")
        return 0.0

    def ex_kurtosis(self):
        if self.scale == 0:
            raise Exception("Moment undefined")
        return 3.0

    def mvsk(self):
        if self.scale == 0:
            raise Exception("Moment undefined")

        mean = self.mean()
        variance = self.variance()
        skewness = self.skewness()
        kurtosis = self.ex_kurtosis()

        return [mean, variance, skewness, kurtosis]

#PARETO


class ParetoDistribution:
    def __init__(self, rand, scale, shape):
        self.rand = rand
        self.scale = scale
        self.shape = shape

    def pdf(self, x):
        if x >= self.scale:
            probability = (self.shape * (self.scale ** self.shape)) / (x ** (self.shape + 1))
            return probability
        else:
            return 0.0

    def cdf(self, x):
        if x >= self.scale:
            probability = 1.0 - (self.scale / x) ** self.shape
            return probability
        else:
            return 0.0

    def ppf(self, p):
        if p < 0 or p > 1:
            raise ValueError("Az input p valószínűségnek 0 és 1 között kell lennie.")

        x = self.scale / ((1 - p) ** (1 / self.shape))
        return x

    def gen_random(self): #nemjo
        rand_gen = self.rand.random()  # Véletlen szám 0 és 1 között
        return self.scale / (rand_gen ** (1 / self.shape))

    def mean(self):
        if self.shape <= 1:
            raise Exception("Moment undefined")

        return (self.shape * self.scale) / (self.shape - 1)

    def variance(self):
        if self.shape <= 2:
            raise Exception("Moment undefined")
        return (self.scale**2 * self.shape) / ((self.shape - 1)**2 * (self.shape - 2))

    def skewness(self):
        if self.shape <= 3:
            raise Exception("Skewness undefined")
        return (2 * (1 + self.shape)) / (self.shape - 3) * ((self.shape - 2) / self.shape) ** 0.5

    def ex_kurtosis(self):
        if self.shape <= 4:
            raise Exception("Excess Kurtosis undefined")
        return 6 * (self.shape ** 3 + self.shape ** 2 - 6 * self.shape - 2) / \
            (self.shape * (self.shape - 3) * (self.shape - 4))

    def mvsk(self):
        if self.shape <= 4:
            raise Exception("Moment undefined")

        mean = self.mean()
        variance = self.variance()
        skewness = self.skewness()
        ex_kurtosis = self.ex_kurtosis()

        return [mean, variance, skewness, ex_kurtosis]

