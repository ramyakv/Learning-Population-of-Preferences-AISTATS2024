from __future__ import annotations
import matplotlib.pyplot as plt
import numpy as np


class UserDistribution:
    _size = 10_000_000
    points: np.ndarray
    lb: int
    ub: int
    d: int
    seed: int

    def __init__(self, lb: int, ub: int, d: int, seed: int) -> UserDistribution:
        self.lb = lb
        self.ub = ub
        self.d = d
        self.seed = seed
        # wants points to be the same, so set the seed to 42 here, then, set to user seed in __post_init__
        self.rng = np.random.default_rng(42)

    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed)
        self.rng.shuffle(self.points)

    def samples(self, size: int, replace: bool) -> np.ndarray:
        samples_idx = self.rng.choice(self._size, size=size, replace=replace)
        return self.points[samples_idx]

    def plot(self):
        _, ax = plt.subplots(1, 1, figsize=(6, 6))

        samples = self.samples(10000, False)
        ax.scatter(samples[:, 0], samples[:, 1])

        ax.set_xlim((self.lb, self.ub))
        ax.set_ylim((self.lb, self.ub))
        plt.show()


class Uniform(UserDistribution):
    def __init__(self, lb: int, ub: int, d: int, seed: int) -> Uniform:
        super().__init__(lb, ub, d, seed)
        self.points = self.rng.uniform(lb, ub, size=(self._size, d)).astype(np.float32)
        super().__post_init__()


class Gaussian(UserDistribution):
    def __init__(self, lb: int, ub: int, d: int, seed: int) -> Gaussian:
        super().__init__(lb, ub, d, seed)
        self.points = self.rng.normal(loc=0, scale=0.1, size=(self._size, d)).astype(
            np.float32
        )
        super().__post_init__()


class Mix2Gaussian(UserDistribution):
    def __init__(self, lb: int, ub: int, d: int, seed: int) -> Mix2Gaussian:
        super().__init__(lb, ub, d, seed)

        n = int(self._size / 2)
        g1 = self.rng.normal(loc=0.5, scale=0.1, size=(n, d)).astype(np.float32)
        g2 = self.rng.normal(loc=-0.5, scale=0.1, size=(n, d)).astype(np.float32)
        self.points = np.vstack([g1, g2])

        super().__post_init__()


class Mix3Gaussian(UserDistribution):
    def __init__(self, lb: int, ub: int, d: int, seed: int) -> Mix3Gaussian:
        super().__init__(lb, ub, d, seed)

        n = int(self._size / 4)
        g1 = self.rng.normal(loc=0.5, scale=0.05, size=(n, d)).astype(np.float32)
        g2 = self.rng.normal(loc=-0.5, scale=0.05, size=(n, d)).astype(np.float32)
        g3 = self.rng.normal(loc=0, scale=0.05, size=(2 * n, d)).astype(np.float32)
        self.points = np.vstack([g1, g2, g3])

        super().__post_init__()


class Mix2Gaussian(UserDistribution):
    def __init__(self, lb: int, ub: int, d: int, seed: int) -> Mix2Gaussian:
        super().__init__(lb, ub, d, seed)

        n = int(self._size / 2)
        g1 = self.rng.normal(loc=0.5, scale=0.1, size=(n, d)).astype(np.float32)
        g2 = self.rng.normal(loc=-0.5, scale=0.1, size=(n, d)).astype(np.float32)
        self.points = np.vstack([g1, g2])

        super().__post_init__()


class TwoPoints(UserDistribution):
    def __init__(self, lb: int, ub: int, d: int, seed: int) -> Mix2Gaussian:
        super().__init__(lb, ub, d, seed)

        
        n = int(self._size / 3)
        g1 = np.full((n, d), 0.5).astype(np.float32)
        g2 = np.full((self._size - n, d), -0.5).astype(np.float32)
        self.points = np.vstack([g1, g2])

        super().__post_init__()

class ThreePoints(UserDistribution):
    def __init__(self, lb: int, ub: int, d: int, seed: int) -> Mix2Gaussian:
        super().__init__(lb, ub, d, seed)

        g1 = np.full((int(self._size * 1/3), d), 0.7).astype(np.float32)
        g2 = np.full((int(self._size * 1/2), d), 0).astype(np.float32)
        g3 = np.full((int(self._size * 1/6), d), -0.7).astype(np.float32)
        self.points = np.vstack([g1, g2, g3])

        super().__post_init__()

class FourPoints(UserDistribution):
    def __init__(self, lb: int, ub: int, d: int, seed: int) -> Mix2Gaussian:
        super().__init__(lb, ub, d, seed)

        g1 = np.full((int(self._size * 0.1), d), 0.9).astype(np.float32)
        g2 = np.full((int(self._size * 0.2), d), 0.3).astype(np.float32)
        g3 = np.full((int(self._size * 0.3), d), -0.3).astype(np.float32)
        g4 = np.full((int(self._size * 0.4), d), -0.9).astype(np.float32)
        self.points = np.vstack([g1, g2, g3, g4])

        super().__post_init__()

if __name__ == "__main__":
    FourPoints(-1, 1, 2, 0).plot()
