import numpy as np
import os, tqdm

# def p(x: np.ndarray, sigma: np.ndarray, N: int=10) -> np.ndarray:
#     p_ = 0
#     for i in tqdm.trange(-N, N + 1):
#         p_ += np.exp(-(x + 2 * np.pi * i) ** 2 / 2 / sigma ** 2)
#     return p_

# def grad(x: np.ndarray, sigma: np.ndarray, N: int=10) -> np.ndarray:
#     p_ = 0
#     for i in tqdm.trange(-N, N + 1):
#         p_ += (x + 2 * np.pi * i) / sigma ** 2 * np.exp(-(x + 2 * np.pi * i) ** 2 / 2 / sigma ** 2)
#     return p_

def score_stable(x_grid: np.ndarray, sigma_grid: np.ndarray, N: int=100) -> np.ndarray:
    """
    Calculates the score function grad(log p(x|sigma)) in a numerically stable way.

    Args:
        x_grid: 1D numpy array of x values.
        sigma_grid: 1D numpy array of sigma values.
        N: The number of terms in the summation (-N to N).

    Returns:
        2D numpy array of shape (len(sigma_grid), len(x_grid)) containing the stable scores.
    """
    x_grid = np.asarray(x_grid).flatten()
    sigma_grid = np.asarray(sigma_grid).flatten()

    x_broadcast = x_grid[None, :]
    sigma_broadcast = sigma_grid[:, None]

    numerator_sum = np.zeros((sigma_grid.shape[0], x_grid.shape[0]), dtype=np.float64)
    denominator_sum = np.zeros((sigma_grid.shape[0], x_grid.shape[0]), dtype=np.float64)

    A_max = -1e38 * np.ones((sigma_grid.shape[0], x_grid.shape[0]), dtype=np.float64)

    for i in tqdm.trange(-N, N + 1):
        term_i = (x_broadcast + 2 * np.pi * i)
        A_i = - np.square(term_i) / (2 * np.square(sigma_broadcast))
        A_max = np.maximum(A_max, A_i)

    for i in tqdm.trange(-N, N + 1):
        term_i = (x_broadcast + 2 * np.pi * i)
        A_i = - np.square(term_i) / (2 * np.square(sigma_broadcast))
        nabla_A_i = - term_i / np.square(sigma_broadcast)

        exp_diff = np.exp(A_i - A_max)

        numerator_sum += exp_diff * nabla_A_i
        denominator_sum += exp_diff

    # denominator_sum += np.finfo(np.float64).eps

    stable_score = numerator_sum / denominator_sum

    return stable_score

def sample(sigma: np.ndarray) -> np.ndarray:
    out = sigma * np.random.randn(*sigma.shape)
    out = (out + np.pi) % (2 * np.pi) - np.pi
    return out

X_MIN, X_N = 1e-5, 5000
SIGMA_MIN, SIGMA_MAX, SIGMA_N = 3e-3, 2, 5000


class Torus:
    def __init__(self, precalc_dir: str, seed: int) -> None:
        self.precalc_dir = precalc_dir
        self.seed = seed
        self.pre_calculate()

    def pre_calculate(self) -> None:
        np.random.seed(self.seed)
        x: np.ndarray = 10 ** np.linspace(np.log10(X_MIN), 0, X_N + 1) * np.pi
        sigma: np.ndarray = 10 ** np.linspace(np.log10(SIGMA_MIN), np.log10(SIGMA_MAX), SIGMA_N + 1) * np.pi

        score_stable_filepath = f'{self.precalc_dir}/.score_stable.npy'
        if os.path.exists(score_stable_filepath):
            print(f"Loading pre-calculated stable score from {score_stable_filepath}")
            self.score_: np.ndarray = np.load(score_stable_filepath)
        else:
            print("Calculating stable score grid...")
            self.score_ = score_stable(x, sigma, N=100)
            print(f"Saving pre-calculated stable score to {score_stable_filepath}")
            np.save(score_stable_filepath, self.score_)

        if os.path.exists(f'{self.precalc_dir}/.score_norm.npy'):
            self.score_norm_ = np.load(f'{self.precalc_dir}/.score_norm.npy')
        else:
            print("Calculating score norm...")
            sigma_samples = sigma[None].repeat(10000, 0).flatten()
            x_samples = sample(sigma_samples)

            score_on_samples = self.score(x_samples, sigma_samples)
            score_on_samples_reshaped = score_on_samples.reshape(10000, -1)
            self.score_norm_: np.ndarray = (score_on_samples_reshaped ** 2).mean(0)

            print(f"Saving pre-calculated score norm to {self.precalc_dir}/.score_norm.npy")
            np.save(f'{self.precalc_dir}/.score_norm.npy', self.score_norm_)

    def score(self, x: np.ndarray, sigma: np.ndarray) -> np.ndarray:
        x = np.asarray(x)
        sigma = np.asarray(sigma)

        x_wrapped = (x + np.pi) % (2 * np.pi) - np.pi

        sign = np.sign(x_wrapped)
        abs_x = np.abs(x_wrapped)
        # abs_x = np.maximum(abs_x, X_MIN * np.pi / 10.0)

        x_scaled = np.log10(abs_x / np.pi)
        x_scaled = (x_scaled - np.log10(X_MIN)) / (0 - np.log10(X_MIN)) * X_N
        x_index = np.round(np.clip(x_scaled, 0, X_N)).astype(int)

        sigma_scaled = np.log10(sigma / np.pi)
        sigma_scaled = (sigma_scaled - np.log10(SIGMA_MIN)) / (np.log10(SIGMA_MAX) - np.log10(SIGMA_MIN)) * SIGMA_N
        sigma_index = np.round(np.clip(sigma_scaled, 0, SIGMA_N)).astype(int)

        return -sign * self.score_[sigma_index, x_index]

    def score_norm(self, sigma: np.ndarray) -> np.ndarray:
        sigma = np.asarray(sigma)

        sigma_scaled = np.log10(sigma / np.pi)
        sigma_scaled = (sigma_scaled - np.log10(SIGMA_MIN)) / (np.log10(SIGMA_MAX) - np.log10(SIGMA_MIN)) * SIGMA_N
        sigma_index = np.round(np.clip(sigma_scaled, 0, SIGMA_N)).astype(int)

        return self.score_norm_[sigma_index]
