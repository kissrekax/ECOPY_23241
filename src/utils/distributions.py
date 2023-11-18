import random
import math
import pyerf
import scipy


# NORMAL
class NormalDistribution:
    def __init__(self, rand, loc, scale):
        self.rand = rand
        self.loc = loc
        self.scale = scale
        self.sd = math.sqrt(self.scale)

    def pdf(self, x):
        exponent = -0.5 * (((x - self.loc) / self.sd) ** 2)
        normalization = 1 / (self.sd * (math.sqrt(2 * math.pi)))
        probability_density = normalization * math.exp(exponent)
        return probability_density

    def cdf(self, x):
        z = (x - self.loc) / (self.sd * math.sqrt(2))
        cumulative_probability = 0.5 * (1 + pyerf.erf(z))
        return cumulative_probability

    def ppf(self, p):
        return self.loc + self.sd*math.sqrt(2)*pyerf.erfinv(2*p - 1)

    def gen_rand(self):
        u = random.gauss(self.loc, self.sd)
        return u

    def mean(self):
        return self.loc

    def median(self):
        return self.loc

    def variance(self):
        if self.scale is None:
            raise Exception("Moment undefined")
        return self.scale ** 2

    def skewness(self):
        return 0

    def ex_kurtosis(self):
        return 0

    def mvsk(self):
        mean = self.loc
        variance = self.scale ** 2
        std_dev = math.sqrt(variance)
        skewness = 0
        kurtosis = 0

        return [mean, std_dev, skewness, kurtosis]


# LOGISTIC
class LogisticDistribution:
    def __init__(self, rand, location, scale):
        self.rand = rand
        self.location = location
        self.scale = scale

    def pdf(self, x):
        exponent = math.exp(-(x - self.location) / self.scale)
        pdf_value = exponent / (self.scale * (1 + exponent) ** 2)
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


# CHISQUARED
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

    def gen_rand(self):
        x = random.uniform(self.a, self.b)
        return x

    def mean(self):
        return (self.a + self.b) / 2

    def median(self):
        return (self.a + self.b) / 2

    def variance(self):
        interval_length = self.b - self.a
        return (interval_length ** 2) / 12

    def skewness(self):
        return 0

    def ex_kurtosis(self):
        return -6 / 5

    def mvsk(self):
        mean = (self.a + self.b) / 2
        variance = ((self.b - self.a) ** 2) / 12
        stdev = math.sqrt(variance)
        skewness = 0
        kurtosis = -6 / 5

        return [mean, variance, skewness, kurtosis]


# CAUCHY
class CauchyDistribution:
    def __init__(self, rand, loc, scale):
        self.rand = rand
        self.loc = loc
        self.scale = scale

    def pdf(self, x):
        if self.scale <= 0:
            raise ValueError("A skála értékének pozitívnak kell lennie.")

        denominator = math.pi * self.scale * (1 + ((x - self.loc) / self.scale) ** 2)
        probability_density = 1 / denominator
        return probability_density

    def cdf(self, x):
        if self.scale <= 0:
            raise ValueError("A skála értékének pozitívnak kell lennie.")

        cumulative_probability = 0.5 + math.atan((x - self.loc) / self.scale) / math.pi
        return cumulative_probability

    def ppf(self, p):
        if self.scale <= 0:
            raise ValueError("A skála értékének pozitívnak kell lennie.")
        if p < 0 or p > 1:
            raise ValueError("A valószínűségnek 0 és 1 között kell lennie.")

        x = self.loc + self.scale * math.tan(math.pi * (p - 0.5))
        return x

    def gen_rand(self):
        if self.scale <= 0:
            raise ValueError("A skála értékének pozitívnak kell lennie.")

        z = random.uniform(0, 1)
        x = self.loc + self.scale * math.tan(math.pi * (z - 0.5))
        return x

    def mean(self):
        raise Exception('Moments undefined')

    def median(self):
        return self.loc

    def variance(self):
        raise Exception('Moments undefined')

    def skewness(self):
        raise Exception('Moments undefined')

    def ex_kurtosis(self):
        raise Exception('Moments undefined')

    def mvsk(self):
        mean = self.mean()
        variance = self.variance()
        skewness = self.skewness()
        ex_kurtosis = self.ex_kurtosis()

        try:
            return [mean, variance, skewness, ex_kurtosis]
        except:
            raise Exception("Moments undefined")


# LAPLACE
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

    def gen_rand(self):  # nem jó
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


# PARETO
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

    def gen_rand(self):  # nem jó
        return random.paretovariate(self.shape)

    def mean(self):
        if self.shape <= 1:
            raise Exception("Moment undefined")

        return (self.shape * self.scale) / (self.shape - 1)

    def variance(self):
        if self.shape <= 2:
            raise Exception("Moment undefined")
        return (self.scale ** 2 * self.shape) / ((self.shape - 1) ** 2 * (self.shape - 2))

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
