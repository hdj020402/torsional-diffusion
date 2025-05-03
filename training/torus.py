import numpy as np
import os, tqdm
from utils.setup_seed import setup_seed

def p(x: np.ndarray, sigma: np.ndarray, N: int=10) -> np.ndarray:
    p_ = 0
    for i in tqdm.trange(-N, N + 1):
        p_ += np.exp(-(x + 2 * np.pi * i) ** 2 / 2 / sigma ** 2)
    return p_

def grad(x: np.ndarray, sigma: np.ndarray, N: int=10) -> np.ndarray:
    p_ = 0
    for i in tqdm.trange(-N, N + 1):
        p_ += (x + 2 * np.pi * i) / sigma ** 2 * np.exp(-(x + 2 * np.pi * i) ** 2 / 2 / sigma ** 2)
    return p_

def sample(sigma: np.ndarray) -> np.ndarray:
    out = sigma * np.random.randn(*sigma.shape)
    out = (out + np.pi) % (2 * np.pi) - np.pi
    return out

X_MIN, X_N = 1e-5, 5000  # relative to pi
SIGMA_MIN, SIGMA_MAX, SIGMA_N = 3e-3, 2, 5000  # relative to pi


class Torus:
    def __init__(self, precalc_dir: str, seed: int) -> None:
        self.precalc_dir = precalc_dir
        self.seed = seed
        self.pre_calculate()

    def pre_calculate(self) -> None:
        setup_seed(self.seed)
        x: np.ndarray = 10 ** np.linspace(np.log10(X_MIN), 0, X_N + 1) * np.pi
        sigma: np.ndarray = 10 ** np.linspace(np.log10(SIGMA_MIN), np.log10(SIGMA_MAX), SIGMA_N + 1) * np.pi

        if os.path.exists(f'{self.precalc_dir}/.p.npy'):
            self.p_ = np.load(f'{self.precalc_dir}/.p.npy')
        else:
            self.p_ = p(x, sigma[:, None], N=100)
            np.save(f'{self.precalc_dir}/.p.npy', self.p_)

        if os.path.exists(f'{self.precalc_dir}/.score.npy'):
            self.score_ = np.load(f'{self.precalc_dir}/.score.npy')
        else:
            self.score_ = grad(x, sigma[:, None], N=100) / self.p_
            np.save(f'{self.precalc_dir}/.score.npy', self.score_)

        if os.path.exists(f'{self.precalc_dir}/.score_norm.npy'):
            self.score_norm_ = np.load(f'{self.precalc_dir}/.score_norm.npy')
        else:
            score_norm_ = self.score(
                sample(sigma[None].repeat(10000, 0).flatten()),
                sigma[None].repeat(10000, 0).flatten()
                ).reshape(10000, -1)
            self.score_norm_: np.ndarray = (score_norm_ ** 2).mean(0)
            np.save(f'{self.precalc_dir}/.score_norm.npy', self.score_norm_)

    def score(self, x: np.ndarray, sigma: np.ndarray) -> np.ndarray:
        x = (x + np.pi) % (2 * np.pi) - np.pi
        sign = np.sign(x)
        x = np.log(np.abs(x) / np.pi)
        x = (x - np.log(X_MIN)) / (0 - np.log(X_MIN)) * X_N
        x = np.round(np.clip(x, 0, X_N)).astype(int)
        sigma = np.log(sigma / np.pi)
        sigma = (sigma - np.log(SIGMA_MIN)) / (np.log(SIGMA_MAX) - np.log(SIGMA_MIN)) * SIGMA_N
        sigma = np.round(np.clip(sigma, 0, SIGMA_N)).astype(int)
        return -sign * self.score_[sigma, x]

    def score_norm(self, sigma: np.ndarray) -> np.ndarray:
        sigma = np.log(sigma / np.pi)
        sigma = (sigma - np.log(SIGMA_MIN)) / (np.log(SIGMA_MAX) - np.log(SIGMA_MIN)) * SIGMA_N
        sigma = np.round(np.clip(sigma, 0, SIGMA_N)).astype(int)
        return self.score_norm_[sigma]
