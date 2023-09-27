#NORMAL

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

