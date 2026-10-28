"""ЛР2, вариант 7–1–4: SMC → зануление → расстояние KS.

Переиспользует уравнение, солвер, граф и загрузчик реальных данных ЛР1.
Импорт не запускает эксперимент и не требует сети.
"""
from dataclasses import dataclass
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import sympy as sp
from scipy.special import logsumexp
from scipy.stats import ks_2samp

import oscillators as L

NAMES = ("k", "c", "b")


@dataclass
class Dataset:
    name: str
    t: np.ndarray
    X: np.ndarray
    y: np.ndarray
    truth: np.ndarray | None = None


def synthetic(damping=0.18, seed=17, n=600):
    """Независимый шум ускорения; x и v известны без ошибок."""
    t = np.linspace(0, 18, n)
    state = L.solve(L.rhs_spring, t, [1.0, 0.0], 2.5, damping / 5)
    X = np.column_stack([-state[:, 0], -state[:, 1], np.ones(n)])
    theta = np.array([6.25, damping, 0.0])
    y = X @ theta + np.random.default_rng(seed).normal(0, 0.08, n)
    return Dataset(f"Синтетика c={damping:g}", t, X, y, theta)


def real_data(cache_dir):
    """Локальная копия в явном кеше: штатный загрузчик ЛР1 проверит SHA256."""
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    target = cache / L.SOURCES["science"]["file"]
    if not target.exists():
        shutil.copyfile(L.LOCAL_DATA / target.name, target)
    x, v, a = L.load_airtrack_derivatives(cache_dir=cache)
    t, _ = L.load_airtrack(cache_dir=cache)
    if len(t) != len(a):
        raise ValueError("Несовместимые таблицы координат и производных")
    return Dataset("Тележка Science 2009", t, np.column_stack([-x, -v, np.ones(len(a))]), a)


def zero_input(X, column, fraction=1.0, seed=42):
    """Зануляет долю строк выбранного наблюдаемого входа, y не меняет.

    Одна перестановка обеспечивает вложенность масок при росте fraction.
    fraction=1 — выбранная интервенция из задания в чистом виде.
    """
    if column not in (0, 1) or not 0 <= fraction <= 1:
        raise ValueError("column: 0 (x) или 1 (v); fraction: от 0 до 1")
    out = np.array(X, float, copy=True)
    ids = np.random.default_rng(seed).permutation(len(out))[:round(fraction * len(out))]
    out[ids, column] = 0
    return out


def prior_scale(X, y):
    # Фиксируем по исходному train, затем НЕ меняем при интервенциях.
    return np.array([20.0, 5.0, max(5 * np.std(y), 1.0)])


def noise_scale(X, y):
    residual = y - X @ np.linalg.lstsq(X, y, rcond=None)[0]
    return max(float(np.sqrt(residual @ residual / (len(y) - X.shape[1]))), 1e-6)


def gaussian_posterior(X, y, sigma, scale):
    """Точный контроль для нормального prior и квадратичной ошибки."""
    cov = np.linalg.inv(np.diag(1 / scale**2) + X.T @ X / sigma**2)
    return cov @ X.T @ y / sigma**2, cov


def systematic_resample(weights, rng):
    positions = (rng.random() + np.arange(len(weights))) / len(weights)
    cumulative = np.cumsum(weights)
    cumulative[-1] = 1.0
    return np.searchsorted(cumulative, positions)


def smc(X, y, sigma, scale, loss="MSE", n_particles=2048, seed=0, moves=10):
    """Адаптивный tempered resample–move SMC в координатах prior N(0,I).

    π_beta ∝ prior * likelihood**beta, beta: 0→1. ESS следующего шага ≥0.8N.
    После каждого взвешивания — systematic resampling и MH rejuvenation.
    MH здесь внутреннее ядро SMC, а не самостоятельный метод варианта 6.
    """
    if loss not in ("MSE", "MAE") or sigma <= 0 or n_particles < 32:
        raise ValueError("Нужны MSE/MAE, sigma>0, N>=32")
    rng = np.random.default_rng(seed)
    A = np.asarray(X) * np.asarray(scale)
    gram, score, yy = A.T @ A, A.T @ y, y @ y

    def likelihood(z):
        if loss == "MSE":
            sse = np.einsum("ni,ij,nj->n", z, gram, z) - 2 * (z @ score) + yy
            return -np.maximum(sse, 0) / (2 * sigma**2)
        # Laplace b=sigma/sqrt(2): тот же масштаб стандартного отклонения.
        out = np.empty(len(z))
        for lo in range(0, len(z), 256):
            residual = y[None, :] - z[lo:lo+256] @ A.T
            out[lo:lo+256] = -np.abs(residual).sum(axis=1) * np.sqrt(2) / sigma
        return out

    z = rng.normal(size=(n_particles, X.shape[1]))
    ll = likelihood(z)
    beta, history = 0.0, []
    for stage in range(200):
        def weights_at(value):
            logw = (value - beta) * ll
            return np.exp(logw - logsumexp(logw))

        lo, hi = beta, 1.0
        if 1 / np.sum(weights_at(hi)**2) < 0.8 * n_particles:
            for _ in range(45):
                mid = (lo + hi) / 2
                if 1 / np.sum(weights_at(mid)**2) < 0.8 * n_particles:
                    hi = mid
                else:
                    lo = mid
            nxt = lo
        else:
            nxt = 1.0
        weights = weights_at(nxt)
        ess = 1 / np.sum(weights**2)
        # Proposal фиксируется до resampling, симметрично для всех частиц.
        center = weights @ z
        covariance = ((z - center).T * weights) @ (z - center)
        chol = np.linalg.cholesky(covariance + np.eye(z.shape[1]) * 1e-14)
        ids = systematic_resample(weights, rng)
        z, ll = z[ids].copy(), ll[ids].copy()
        beta = nxt
        lp = -0.5 * np.sum(z*z, axis=1) + beta * ll
        accepts = []
        for _ in range(moves + (20 if beta == 1 else 0)):
            proposed = z + (rng.normal(size=z.shape) @ chol.T) * (2.38 / np.sqrt(z.shape[1]))
            proposed_ll = likelihood(proposed)
            proposed_lp = -0.5 * np.sum(proposed**2, axis=1) + beta * proposed_ll
            accept = np.log(rng.random(n_particles)) < proposed_lp - lp
            z[accept], ll[accept], lp[accept] = proposed[accept], proposed_ll[accept], proposed_lp[accept]
            accepts.append(np.mean(accept))
        history.append(dict(beta=beta, ess=ess, acceptance=np.mean(accepts)))
        if beta == 1:
            return z * scale, pd.DataFrame(history)
    raise RuntimeError("SMC не дошёл до beta=1 за 200 шагов")


def ks_distance(left, right):
    """Маргинальные D; p-value намеренно не используется для зависимых частиц."""
    return np.array([ks_2samp(left[:, j], right[:, j]).statistic for j in range(left.shape[1])])


def expression_graph():
    # k=w², c=2*zeta*w: переобозначение коэффициентов уравнения ЛР1 + nuisance b.
    k, c, b, x, v = sp.symbols("k c b x v")
    return L.expression_graph({"a": -k*x - c*v + b})


def experiment(data, n_particles=2048, seed=100, losses=("MSE", "MAE"), controls=5):
    split = int(0.7 * len(data.y))
    X, y = data.X[:split], data.y[:split]
    Xtest, ytest = data.X[split:], data.y[split:]
    scale, sigma = prior_scale(X, y), noise_scale(X, y)
    result = dict(data=data, split=split, scale=scale, sigma=sigma, samples={}, diagnostics={}, curves=[])
    rows, checks, noise_rows = [], [], []
    for li, loss in enumerate(losses):
        base, hist = smc(X, y, sigma, scale, loss, n_particles, seed + 100*li)
        result["samples"][loss, "original"] = base
        result["diagnostics"][loss, "original"] = hist
        null = []
        for rep in range(controls):
            repeated, _ = smc(X, y, sigma, scale, loss, n_particles, seed + 100*li + rep + 1)
            d = ks_distance(base, repeated)
            null.append(d.max())
            noise_rows.append(dict(dataset=data.name, loss=loss, repeat=rep, D_max=d.max()))
        threshold = max(0.10, float(np.quantile(null, 0.95)))
        mean = np.mean(base, axis=0)
        rmse = np.sqrt(np.mean((ytest - Xtest @ mean)**2))
        if loss == "MSE":
            exact_mean, exact_cov = gaussian_posterior(X, y, sigma, scale)
            for j, name in enumerate(NAMES):
                checks.append(dict(dataset=data.name, parameter=name,
                    mean_error_sd=abs(mean[j]-exact_mean[j])/np.sqrt(exact_cov[j,j]),
                    sd_ratio=np.std(base[:,j])/np.sqrt(exact_cov[j,j])))
        for j, variable in enumerate(("x", "v")):
            changed, hist = smc(zero_input(X, j), y, sigma, scale, loss, n_particles, seed+100*li+20+j)
            result["samples"][loss, variable] = changed
            result["diagnostics"][loss, variable] = hist
            d = ks_distance(base, changed)
            # Обнуление на train эквивалентно удалению коэффициента; прочие переоцениваем.
            reduced_mean = changed.mean(axis=0)
            reduced_mean[j] = 0
            reduced_rmse = np.sqrt(np.mean((ytest - Xtest @ reduced_mean)**2))
            delta = reduced_rmse / rmse - 1
            effect = np.quantile(np.abs(base[:,j]) * np.std(X[:,j]) / np.std(y), 0.95)
            # Порог 1% для вклада в ускорение, 5% для изменения test RMSE.
            removable = effect < 0.01 and delta <= 0.05
            for q, parameter in enumerate(NAMES):
                rows.append(dict(dataset=data.name, loss=loss, intervention=variable,
                    parameter=parameter, D=d[q], threshold=threshold, shift=d[q]>threshold,
                    baseline_mean=mean[q], baseline_lo=np.quantile(base[:,q],.025),
                    baseline_hi=np.quantile(base[:,q],.975), intervention_mean=changed[:,q].mean(),
                    effect95=effect, test_rmse=rmse, reduced_rmse=reduced_rmse,
                    relative_rmse=delta, decision="удалить" if removable else "сохранить"))
            if loss == "MSE":
                for fraction in (0, .25, .5, .75, 1):
                    if fraction == 0:
                        distance = np.zeros(3)
                    elif fraction == 1:
                        distance = d
                    else:
                        particles, _ = smc(zero_input(X, j, fraction), y, sigma, scale,
                                           loss, n_particles, seed+30+j)
                        distance = ks_distance(base, particles)
                    for q, parameter in enumerate(NAMES):
                        result["curves"].append(dict(variable=variable, fraction=fraction,
                                                     parameter=parameter, D=distance[q]))
    result.update(table=pd.DataFrame(rows), checks=pd.DataFrame(checks),
                  noise=pd.DataFrame(noise_rows), curves=pd.DataFrame(result["curves"]))
    return result


def plot_result(result, output_dir):
    """Графики возвращаются в notebook и сохраняются для отчёта/презентации."""
    import matplotlib.pyplot as plt
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    figs = []
    data = result["data"]
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.2))
    for j, label in enumerate(("x", "v", "a")):
        values = -data.X[:,j] if j < 2 else data.y
        axs[j].plot(data.t, values, lw=.8)
        axs[j].axvline(data.t[result["split"]], color="red", ls="--", label="train / test")
        axs[j].set(xlabel="t (ед. времени источника)", ylabel=label)
    axs[0].legend()
    fig.suptitle(data.name)
    fig.tight_layout()
    fig.savefig(output / "data.png", dpi=140)
    figs.append(fig)
    # Масштаб prior после удаления на порядки шире исходного posterior:
    # отдельная детализация нужна, чтобы форма исходных оценок была видна.
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.3))
    for j, name in enumerate(NAMES):
        for loss in sorted({key[0] for key in result["samples"]}):
            values = result["samples"][loss,"original"][:,j]
            axs[j].hist(values, bins=45, density=True, histtype="step", label=loss)
        if data.truth is not None:
            axs[j].axvline(data.truth[j], ls=":", color="black", label="истинное")
        axs[j].set(xlabel=name, ylabel="плотность", title="Исходный posterior, увеличенный масштаб")
        axs[j].legend()
    fig.tight_layout()
    fig.savefig(output / "original_zoom.png", dpi=140)
    figs.append(fig)
    for loss in sorted({key[0] for key in result["samples"]}):
        fig, axs = plt.subplots(2, 3, figsize=(13, 6))
        for j, name in enumerate(NAMES):
            original = result["samples"][loss,"original"][:,j]
            # Центральные 99.5% всех распределений, единая сетка по строкам.
            all_values = np.concatenate([result["samples"][loss,k][:,j] for k in ("original","x","v")])
            bins = np.linspace(*np.quantile(all_values,[.0025,.9975]), 65)
            for key, label in (("original","исходные"),("x","x → 0"),("v","v → 0")):
                a = result["samples"][loss,key][:,j]
                axs[0,j].hist(a, bins=bins, density=True, histtype="step", label=label)
                sorted_a = np.sort(a)
                axs[1,j].step(sorted_a, np.arange(1,len(a)+1)/len(a), label=label)
            if data.truth is not None:
                axs[0,j].axvline(data.truth[j], ls=":", color="black", label="истинное")
            axs[0,j].set(title=f"{name}, {loss}", ylabel="плотность")
            axs[1,j].set(xlabel=name, ylabel="ECDF")
        axs[0,0].legend(fontsize=8)
        fig.suptitle("Полное зануление: коэффициент потерянного входа возвращается к prior")
        fig.tight_layout()
        fig.savefig(output / f"posterior_{loss}.png", dpi=140)
        figs.append(fig)
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.5))
    for j, variable in enumerate(("x", "v")):
        for name in NAMES:
            q = result["curves"].query("variable == @variable and parameter == @name")
            axs[j].plot(q.fraction, q.D, "o-", label=name)
        axs[j].set(title=f"Зануление {variable}", xlabel="Доля занулённых строк", ylabel="D KS", ylim=(-.02,1.02))
        axs[j].legend()
    fig.tight_layout()
    fig.savefig(output / "dose.png", dpi=140)
    figs.append(fig)
    return figs
