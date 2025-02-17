class Stats:
    def __init__(self, count, mean, std, min, max, percentile_25, percentile_50, percentile_75):
        self.count = count
        self.mean = mean
        self.std = std
        self.min = min
        self.max = max
        self.percentile_25 = percentile_25
        self.percentile_50 = percentile_50
        self.percentile_75 = percentile_75

    def __repr__(self):
        return (
            f"Count {self.count}\n"
            f"Mean {self.mean}\n"
            f"Std {self.std}\n"
            f"Min {self.min}\n"
            f"25% {self.percentile_25}\n"
            f"50% {self.percentile_50}\n"
            f"75% {self.percentile_75}\n"
            f"Max {self.max}\n"
        )
