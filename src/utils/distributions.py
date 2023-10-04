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

class ChiSquaredDistribution:
    def __init__(self, rand, dof):

        self.rand = rand
        self.dof = dof

    def pdf(self, k):
        if self.dof < 0:
            return 0  # PDF is defined for x >= 0

        numerator = self.dof ** ((k / 2) - 1) * math.exp(-self.dof / 2)
        denominator = 2 ** (k / 2) * math.gamma(k / 2)

        pdf = numerator / denominator

        return pdf

    def cdf(self, x):

        cdf_value = chisquare.cdf(x, df=self.dof)
        return cdf_value

    def ppf(self, p):

        ppf_value = chisquare.ppf(p, df=self.dof)
        return ppf_value

    def gen_rand(self):
        return self.rand.logistic()

    def mean(self):

        if self.dof <= 1:
            raise Exception("Moment undefined")

        mean_value = self.dof
        return mean_value

    def variance(self):
        if self.dof <= 1:
            raise Exception("Moment undefined")
        return 2 * self.dof

    def skewness(self):
        if self.dof <= 1:
            raise Exception("Moment undefined")

        return math.sqrt(8.0 / self.dof)

    def ex_kurtosis(self):

        if self.dof <= 1:
            raise Exception("Moment undefined")

        ex_kurtosis_value = 12.0 / self.dof
        return ex_kurtosis_value

    def mvsk(self):
        if self.dof == 0:
            raise Exception("Moment undefined")

        mean = self.mean()
        variance = self.variance()
        skewness = self.skewness()
        kurtosis = self.ex_kurtosis()

        return [mean, variance, skewness, kurtosis]




















