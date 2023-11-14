#NORMAL

import random
import math
from scipy.stats import chisquare

class NormalDistribution:
    def __init__(self, rand, loc, scale):
        self.rand = rand
        self.loc = loc
        self.scale = scale

    def pdf(self, x):
        if self.scale <= 0:
            raise ValueError("A szórásnak pozitívnak kell lennie.")

        exponent = -((x - self.loc) ** 2) / (2 * self.scale ** 2)
        normalization = 1 / (self.scale * math.sqrt(2 * math.pi))
        probability_density = normalization * math.exp(exponent)
        return probability_density

    def cdf(self, x):
        if self.scale <= 0:
            raise ValueError("A szórásnak pozitívnak kell lennie.")

        z = (x - self.loc) / (self.scale * math.sqrt(2))
        cumulative_probability = 0.5 * (1 + math.erf(z))
        return cumulative_probability

    def ppf(self, p):
        if self.scale <= 0:
            raise ValueError("A szórásnak pozitívnak kell lennie.")
        if p < 0 or p > 1:
            raise ValueError("A valószínűségnek 0 és 1 között kell lennie.")

        z = math.erfinv(2 * p - 1) * math.sqrt(2)
        x = z * self.scale + self.loc
        return x

    def gen_random(self):
        if self.scale <= 0:
            raise ValueError("A szórásnak pozitívnak kell lennie.")

        z = random.normalvariate(0, 1)
        x = z * self.scale + self.loc
        return x

    def mean(self):
        return self.loc

    def median(self):
        return self.loc

    def variance(self):
        if self.scale is None:
            raise Exception("Moment undefined")
        return self.scale ** 2

    def skewness(self):
        return 0  # A normális eloszlás ferdesége mindig 0

    def ex_kurtosis(self):
        return 0  # A normális eloszlás többlet csúcsossága mindig 0

    def mvsk(self):
        mean = self.loc
        variance = self.scale ** 2
        std_dev = math.sqrt(variance)
        skewness = 0  # Normális eloszlásnál a ferdeség értéke mindig 0
        kurtosis = 0  # Normális eloszlásnál a többlet csúcsosság értéke mindig 0

        return [mean, std_dev, skewness, kurtosis]

#LOGISTIC

class LogisticDistribution:
    def __init__(self, rand, location, scale):
        self.rand = rand
        self.location = location
        self.scale = scale

    def pdf(self, x):
        exponent = math.exp(-(x - self.location) / self.scale)
        pdf_value = exponent / (self.scale * (1 + exponent)**2)
        return pdf_value

    def cdf(self, x):
        cdf_value = 1 / (1 + math.exp(-(x - self.location) / self.scale))
        return cdf_value

    def ppf(self, p):

        if p < 0 or p > 1:
            raise ValueError("p értéke csak 0 és 1 között lehet.")

        ppf_value = self.location - self.scale * math.log(1 / p - 1)
        return ppf_value

    def gen_rand(self):
        u = self.rand.random()
        rand_value = self.location + self.scale * math.log(u / (1 - u))
        return rand_value

    def mean(self):
        if self.scale == 0:
            raise Exception("Moment undefined")

        mean_value = self.location
        return mean_value

    def variance(self):
        if self.scale == 0:
            raise Exception("Moment undefined")

        variance_value = (math.pi ** 2 * self.scale ** 2) / 3
        return variance_value

    def skewness(self):
        if self.scale == 0:
            raise Exception("Moment undefined")

        skewness_value = 0
        return skewness_value

    def ex_kurtosis(self):
        if self.scale == 0:
            raise Exception("Moment undefined")

        ex_kurtosis_value = 1.2
        return ex_kurtosis_value

    def mvsk(self):
        if self.scale == 0:
            raise Exception("Moment undefined")

        mean = self.mean()
        variance = self.variance()
        skewness = self.skewness()
        kurtosis = self.ex_kurtosis()

        return [mean, variance, skewness, kurtosis]


#CHISQUARED

import random
import typing
import math
import scipy.special

class ChiSquaredDistribution:
    def __init__(self, rand, dof):
        self.rand = rand
        self.dof = dof

    def pdf(self, x):
        if x < 0:
            return 0
        numerator = x ** ((self.dof / 2) - 1) * math.exp(-x / 2)
        denominator = (2 ** (self.dof / 2)) * scipy.special.gamma(self.dof / 2)
        return numerator / denominator

    def cdf(self, x):
        if x < 0:
            return 0
        return scipy.special.gammainc(self.dof / 2, x / 2)

    def ppf(self, p):
        if p < 0 or p > 1:
            raise ValueError("p must be in the range [0, 1]")
        return 2 * scipy.special.gammaincinv(self.dof / 2, p)

    def gen_rand(self):
        u = self.rand.random()
        return self.ppf(u)

    def mean(self):
        try:
            return self.dof
        except:
            raise ValueError("Moment undefined")

    def variance(self):
        return 2 * self.dof

    def skewness(self):
        return math.sqrt(8 / self.dof)

    def ex_kurtosis(self):
        return 12 / self.dof

    def mvsk(self):
        mean = self.mean()
        variance = self.variance()
        skewness = self.skewness()
        ex_kurtosis = self.ex_kurtosis()
        try:
            return [mean, variance, skewness, ex_kurtosis]
        except:
            raise ValueError("Moment undefined")


# UNIFORM

import random

class UniformDistribution:
    def __init__(self, rand, a, b):
        self.rand = rand
        self.a = a
        self.b = b

    def pdf(self, x):
        if self.a >= self.b:
            raise ValueError("Az alsó határnak kisebbnek kell lennie, mint a felső határ.")

        if x < self.a or x > self.b:
            return 0
        else:
            interval_length = self.b - self.a
            probability_density = 1 / interval_length
            return probability_density

    def cdf(self, x):
        if self.a >= self.b:
            raise ValueError("Az alsó határnak kisebbnek kell lennie, mint a felső határ.")

        if x < self.a:
            return 0
        elif x >= self.b:
            return 1
        else:
            interval_length = self.b - self.a
            cumulative_probability = (x - self.a) / interval_length
            return cumulative_probability

    def ppf(self, p):
        if self.a >= self.b:
            raise ValueError("Az alsó határnak kisebbnek kell lennie, mint a felső határ.")
        if p < 0 or p > 1:
            raise ValueError("A valószínűségnek 0 és 1 között kell lennie.")

        x = self.a + p * (self.b - self.a)
        return x

    def gen_random(self):
        if self.a >= self.b:
            raise ValueError("Az alsó határnak kisebbnek kell lennie, mint a felső határ.")

        x = random.uniform(self.a, self.b)
        return x

    def mean(self):
        return (self.a + self.b) / 2

    def median(self):
        return (self.a + self.b) / 2

    def variance(self):
        if self.a >= self.b:
            raise ValueError("Az alsó határnak kisebbnek kell lennie, mint a felső határ.")
        interval_length = self.b - self.a
        return interval_length ** 2 / 12

    def skewness(self):
        return 0  # A uniform eloszlás ferdesége mindig 0

    def ex_kurtosis(self):
        return -6 / 5  # A uniform eloszlás többlet csúcsossága mindig -6/5

    def mvsk(self):
        mean = (self.a + self.b) / 2
        variance = (self.b - self.a) ** 2 / 12
        std_dev = (self.b - self.a) / (12 ** 0.5)
        skewness = 0  # Uniform eloszlásnál a ferdeség értéke mindig 0
        kurtosis = -6 / 5  # Uniform eloszlásnál a többlet csúcsosság értéke mindig -6/5

        return [mean, std_dev, skewness, kurtosis]

# CAUCHY

import math
import random

class CauchyDistribution:
    def __init__(self, rand, location, scale):
        self.rand = rand
        self.location = location
        self.scale = scale

    def pdf(self, x):
        if self.scale <= 0:
            raise ValueError("A skála értékének pozitívnak kell lennie.")

        denominator = math.pi * self.scale * (1 + ((x - self.location) / self.scale) ** 2)
        probability_density = 1 / denominator
        return probability_density

    def cdf(self, x):
        if self.scale <= 0:
            raise ValueError("A skála értékének pozitívnak kell lennie.")

        cumulative_probability = 0.5 + math.atan((x - self.location) / self.scale) / math.pi
        return cumulative_probability

    def ppf(self, p):
        if self.scale <= 0:
            raise ValueError("A skála értékének pozitívnak kell lennie.")
        if p < 0 or p > 1:
            raise ValueError("A valószínűségnek 0 és 1 között kell lennie.")

        x = self.location + self.scale * math.tan(math.pi * (p - 0.5))
        return x

    def gen_random(self):
        if self.scale <= 0:
            raise ValueError("A skála értékének pozitívnak kell lennie.")

        z = random.uniform(0, 1)
        x = self.location + self.scale * math.tan(math.pi * (z - 0.5))
        return x

    def mean(self):
        return float('nan')  # A Cauchy-eloszlásnak nincs meghatározott várható értéke

    def median(self):
        return self.location

    def variance(self):
        return float('inf')  # A Cauchy-eloszlásnak végtelen a varianciája

    def skewness(self):
        return float('nan')  # A Cauchy-eloszlásnak nincs meghatározott ferdesége

    def ex_kurtosis(self):
        return float('nan')  # A Cauchy-eloszlásnak nincs meghatározott többlet csúcsossága

    def mvsk(self):
        mean = float('nan')
        variance = float('inf')
        std_dev = float('inf')
        skewness = float('nan')
        kurtosis = float('nan')

        return [mean, std_dev, skewness, kurtosis]



















