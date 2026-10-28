"""
oscillators.py -- минимальная лабораторная: три колебательные системы.

    1. груз на пружине           x'' = -w² x - 2 zeta w x'
    2. маятник (без пружины)     θ'' = -w0² sin θ - gamma θ'
    3. упругий маятник           r'' = -ws² (|r| - l0) r/|r| - gamma r' - g e_z
       (груз на пружине, который ещё и качается)
    4. маятник Уилберфорса       z'' = -wz² z - bz θ - gamma z',   θ'' = -wθ² θ - bθ z - gamma θ'
       (груз на спиральной пружине: прыгает и поворачивается, движения связаны)

Разделы файла идут в порядке пунктов задания:

    1-2. Уравнения и их вариации, в 3D + время
    3.   Уравнения в виде графа (sympy -> networkx)
    4.   Данные из нескольких независимых источников (загрузка по HTTP)
    5.   Векторы параметров с ожидаемыми значениями
    6-8. Солвер (DOP853), оптимизатор (TRF), подбор параметров, ошибка

Откуда уравнения и вариации (дословно из статей):
    пружина  -- M. Schmidt, H. Lipson, Science 324, 81-85 (2009), доп. материалы, Table S1:
                уравнение движения "a - 0.1·v + 3·x" (есть слагаемое со скоростью = трение)
                и закон энергии "x² + 0.3·v²" (без трения); раздел о тележке: «гармонический
                осциллятор с небольшим затуханием от воздуха и пружин»;
    маятник  -- там же, Table S1: "α - 9.8·sin(θ)"; в тексте: программа находила и
                приближение «x вместо sin(x)» (малые углы), а полный sin -- только на данных
                с размахом больше ~±40°;
    упругий маятник -- P. Lynch, Int. J. Non-Linear Mechanics 37, 345-367 (2002):
                полные уравнения (3)-(5) и линейные (14)-(15);
    маятник Уилберфорса -- R. E. Berg, T. S. Marshall, Am. J. Phys. 59, 32-37 (1991),
                doi:10.1119/1.16702: m z'' + k z + ε/2·θ = 0, I θ'' + δ θ + ε/2·z = 0;
                вариации: без связи (ε = 0) и со связью.
"""
from __future__ import annotations

import hashlib
import io
import os
import urllib.request
import warnings
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import sympy as sp
from scipy import signal, sparse
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares
from scipy.signal import savgol_filter

G = 9.80665                       # ускорение свободного падения, м/с²


# ======================================================================================
# 1-2. Уравнения и их вариации, в 3D
# ======================================================================================
def rhs_spring(t, state, w, zeta):
    """Груз на пружине, любое число осей. state = (положения..., скорости...).

    Для каждой оси: a = -w² r - 2 zeta w v.  Вариация «без трения»: zeta = 0.
    """
    n = len(state) // 2
    r, v = state[:n], state[n:]
    return np.concatenate([v, -w**2 * r - 2 * zeta * w * v])


def rhs_pendulum(t, state, w0, gamma, linear=False):
    """Маятник, угол θ от вертикали. state = (θ, θ').

    θ'' = -w0² sin θ - gamma θ',   w0 = √(g/L).
    Вариация «малые углы» (linear=True): sin θ ≈ θ -- тогда это уравнение пружины.
    """
    th, om = state
    return np.array([om, -w0**2 * (th if linear else np.sin(th)) - gamma * om])


def rhs_pendulum_3d(t, state, L, gamma, g=G):
    """Тот же маятник в 3D (сферический маятник): груз на нити длины L, вектор r = (x, y, z).

    a = g_vec - r (g_vec·r + |v|²)/L² - gamma v,   g_vec = (0, 0, -g).
    Второе слагаемое -- натяжение нити: оно держит груз на сфере |r| = L.
    В плоскости xz это в точности θ'' = -(g/L) sin θ - gamma θ' (проверено в тестах).
    """
    r, v = state[:3], state[3:]
    gv = np.array([0.0, 0.0, -g])
    a = gv - r * (gv @ r + v @ v) / L**2 - gamma * v
    return np.concatenate([v, a])


def rhs_elastic(t, state, ws, l0, gamma, g=G, linear=False):
    """Упругий маятник в 3D: груз на пружине, точка подвеса в начале координат, z -- вверх.

    Полное уравнение (Lynch 2002, уравнения 3-5):
        a = -ws² (|r| - l0) r/|r| - g e_z - gamma v,   ws = √(k/m), l0 -- длина пружины без груза.
    Пружина тянет к подвесу пропорционально растяжению, тяжесть -- вниз, трение тормозит.

    Вариация «линейная» (linear=True, Lynch 2002, уравнения 14-15): малые колебания около
    положения равновесия, где пружина длиной l = l0 + g/ws². Тогда движения разделяются:
        x'' = -(g/l) x,  y'' = -(g/l) y   -- обычный маятник длины l,
        z'' = -ws² (z + l)                 -- обычная пружина.
    Трение (gamma) в статье нет, мы добавили его сами; gamma = 0 -- как в статье.
    """
    r, v = state[:3], state[3:]
    if linear:
        leq = l0 + g / ws**2
        a = np.array([-g / leq * r[0], -g / leq * r[1], -ws**2 * (r[2] + leq)]) - gamma * v
        return np.concatenate([v, a])
    rho = np.sqrt(r @ r)
    a = -ws**2 * (rho - l0) * r / rho - gamma * v
    a[2] -= g
    return np.concatenate([v, a])


def rhs_wilberforce(t, state, wz, wt, bz, bt, gamma):
    """Маятник Уилберфорса (Berg & Marshall 1991). state = (z, θ, z', θ').

        z'' = -wz² z - bz θ - gamma z'      прыжки: пружина + толчок от поворота
        θ'' = -wθ² θ - bθ z - gamma θ'      повороты: закрутка + толчок от растяжения

    wz = √(k/m), wθ = √(δ/I), bz = ε/(2m), bθ = ε/(2I): ε -- сила связи (спиральная пружина при
    растяжении чуть раскручивается). Вариация «без связи»: bz = bθ = 0 -- два независимых
    осциллятора. Трение gamma добавили мы (в статье его нет; gamma = 0 -- как в статье).
    """
    z, th, vz, vt = state
    return np.array([vz, vt, -wz**2 * z - bz * th - gamma * vz, -wt**2 * th - bt * z - gamma * vt])


def wilberforce_matrix(wz, wt, bz, bt, gamma) -> np.ndarray:
    """То же уравнение в виде state' = A · state (оно линейное)."""
    return np.array([[0, 0, 1, 0], [0, 0, 0, 1],
                     [-wz**2, -bz, -gamma, 0], [-bt, -wt**2, 0, -gamma]], float)


def wilberforce_modes(wz, wt, bz, bt) -> tuple[float, float]:
    """Две «общие» частоты (нормальные моды) связанной системы, рад/с:
    ω² = (wz² + wθ²)/2 ± √(((wθ² - wz²)/2)² + bz·bθ)."""
    a = (wz**2 + wt**2) / 2
    d = np.sqrt(((wt**2 - wz**2) / 2) ** 2 + bz * bt)
    return float(np.sqrt(a + d)), float(np.sqrt(max(a - d, 0.0)))


def wilberforce_point_3d(z, th, rho):
    """Точка на грузе на расстоянии rho от оси пружины: (rho·cos θ, rho·sin θ, z) -- 3D-движение."""
    return np.column_stack([rho * np.cos(th), rho * np.sin(th), z])


# ======================================================================================
# 3. Уравнения в виде графа: формула sympy -> граф networkx
# ======================================================================================
def symbolic_equations() -> dict[str, dict[str, sp.Expr]]:
    """Все четыре уравнения в символьном виде (sympy): {система: {выход: формула}}."""
    x, y, z, vx, vy, vz, th, om = sp.symbols("x y z v_x v_y v_z theta v_theta")
    w, zeta, w0, gamma, ws, l0, g = sp.symbols("w zeta w_0 gamma w_s l_0 g", positive=True)
    wz_, wt_ = sp.symbols("w_z w_theta", positive=True)
    bz_, bt_ = sp.symbols("b_z b_theta", real=True)
    rho = sp.sqrt(x**2 + y**2 + z**2)
    return {
        "пружина": {f"a_{c}": -w**2 * q - 2 * zeta * w * v
                    for c, q, v in (("x", x, vx), ("y", y, vy), ("z", z, vz))},
        "маятник": {"theta''": -w0**2 * sp.sin(th) - gamma * om},
        "упругий маятник": {
            "a_x": -ws**2 * (rho - l0) * x / rho - gamma * vx,
            "a_y": -ws**2 * (rho - l0) * y / rho - gamma * vy,
            "a_z": -ws**2 * (rho - l0) * z / rho - gamma * vz - g,
        },
        "маятник Уилберфорса": {
            "z''": -wz_**2 * z - bz_ * th - gamma * vz,
            "theta''": -wt_**2 * th - bt_ * z - gamma * om,
        },
    }


_OPS = {sp.Add: "+", sp.Mul: "×", sp.Pow: "^", sp.sin: "sin"}


def expression_graph(outputs: dict[str, sp.Expr]) -> nx.DiGraph:
    """Граф формул: узел = переменная, параметр, число или операция; ребро «вход -> операция».

    Одинаковые части формул (например, |r| у упругого маятника) становятся одним общим узлом:
    sympy хранит одинаковые подвыражения как один объект.
    """
    G_ = nx.DiGraph()

    def add(e) -> str:
        key = sp.srepr(e)
        if key in G_:
            return key
        if e.is_Symbol:
            G_.add_node(key, kind="symbol", label=sp.pretty(e), expr=e)
        elif e.is_Number:
            G_.add_node(key, kind="number", label=str(e), value=float(e))
        else:
            op = _OPS[type(e)]
            G_.add_node(key, kind="op", label=op, op=op)
            for i, arg in enumerate(e.args):
                G_.add_edge(add(arg), key, pos=i)
        return key

    for name, e in outputs.items():
        G_.add_node(name, kind="output", label=name)
        G_.add_edge(add(e), name, pos=0)
    return G_


def evaluate_graph(G_: nx.DiGraph, values: dict) -> dict:
    """Считает граф: обход узлов в топологическом порядке (входы раньше результата).

    values -- {имя символа: число или массив}. Возвращает {имя выхода: значение}.
    """
    val = {}
    for node in nx.topological_sort(G_):
        d = G_.nodes[node]
        args = [val[p] for p in sorted(G_.predecessors(node), key=lambda p: G_.edges[p, node]["pos"])]
        if d["kind"] == "symbol":
            val[node] = values[str(d["expr"])]
        elif d["kind"] == "number":
            val[node] = d["value"]
        elif d["kind"] == "output":
            val[node] = args[0]
        elif d["op"] == "+":
            val[node] = sum(args[1:], args[0])
        elif d["op"] == "×":
            val[node] = np.prod(np.broadcast_arrays(*args), axis=0) if len(args) > 1 else args[0]
        elif d["op"] == "^":
            val[node] = np.power(np.asarray(args[0], float), args[1])
        elif d["op"] == "sin":
            val[node] = np.sin(args[0])
    return {n: val[n] for n in G_.nodes if G_.nodes[n]["kind"] == "output"}


def graph_layout(G_: nx.DiGraph) -> dict:
    """Узлы слева направо: столбец = сколько шагов от входов."""
    depth = {}
    for node in nx.topological_sort(G_):
        preds = list(G_.predecessors(node))
        depth[node] = 0 if not preds else 1 + max(depth[p] for p in preds)
    columns: dict[int, list] = {}
    for node, d in depth.items():
        columns.setdefault(d, []).append(node)
    return {node: (d, -i * 10 / max(len(nodes) - 1, 1))
            for d, nodes in columns.items() for i, node in enumerate(nodes)}


# ======================================================================================
# 4. Данные: загрузка по HTTP с проверкой контрольной суммы
# ======================================================================================
_GH = "https://raw.githubusercontent.com"
_EL = f"{_GH}/edwardwawrzynek/elastic_pendulum/7ac1f6b13a17c01caa36b06f4c0266e578259da7/data"

SOURCES = {
    "science": {
        "что": "Schmidt & Lipson, Science 2009, доп. материалы: тележка на пружинах "
               "(real_linear_*) и маятник (real_pend_*), видеотрекинг Vicon",
        "url": "https://www.science.org/doi/suppl/10.1126/science.1165893/suppl_file/invar_datasets.zip",
        "file": "invar_datasets.zip",
        "sha256": "9794ee02f45403f87b739cc235a046bd424a73b8222067165f66463cef57be59",
    },
    "video": {
        "что": "J. Chiang, University of Toronto (PHY180): маятник на нити, видеотрекинг OpenCV, "
               "координаты в пикселях, 12 кадров/с",
        "url": f"{_GH}/supreme-gg-gg/pendulum-tracking/57f311732447a97d0d0a276bd93f67670b2de298/sample/positions.csv",
        "file": "positions.csv",
        "sha256": "f1fe8d272fc478fcf42a6b34ae47be781ac85d4a09cff2f136b56bba80138748",
    },
    **{f"elastic_{m}g": {
        "что": f"E. Wawrzynek (GitHub): упругий маятник, груз {m} г, видео 30 кадров/с, Tracker",
        "url": f"{_EL}/m{m}g/track.csv",
        "file": f"elastic_m{m}g.csv",
        "sha256": h,
    } for m, h in ((400, "2b35167b867e5f91ce9932217666159f57b2dd247a77285aad71df478c04697d"),
                   (500, "f5b096b29989e699fdf6c93b7f5aba92b6dbea53e7e5cf5ce4af660fc9ef6420"),
                   (700, "057012d1ef1fd22fd594297ff7550faa597209889b1cfaba8a917b343fd4e1c3"),
                   (900, "510685e3cb39b2bb251d84279e479fa919c1337fa1ffa7e69ff82a10471f5093"),
                   (1000, "08cf1ed3d56ebb9aec700e8b7aadec1bddb036015c113d5501ff37d284a24eaa"))},
}
_WB = f"{_GH}/RodrigoGarciaTejera/WilberforcePendulum/34f24dfe1766f549b1f1b6a1f01d11949816e326"
_MAR = f"{_GH}/emarinhojr/wilberforce_pendulum_plots/56edfda3eedfd0c99d37da6f1a84fcaeb9c8a515"
SOURCES.update({
    "wilberforce_phone": {
        "что": "Gallot, Gau, García-Tejera, arXiv:2212.06949 (2022): маятник Уилберфорса, груз -- смартфон; "
               "ускорение и гироскоп по 3 осям, 10 измерений/с (запись bat1 -- биения)",
        "url": f"{_WB}/bat1.csv",
        "file": "wilberforce_bat1.csv",
        "sha256": "cea2bd2d9312945dd9296b263e1759d0093906318b19868e9e3548699c13d1ed",
    },
    "wilberforce_video_z": {
        "что": "E. Marinho Jr. (GitHub): маятник Уилберфорса, видео 30 кадров/с, высота груза z (м)",
        "url": f"{_MAR}/z_data.csv",
        "file": "wilberforce_z_data.csv",
        "sha256": "3b2bb3a8d150219cce2c636b83f12aea2675c0a9c959d0a67687c8713c046582",
    },
    "wilberforce_video_theta": {
        "что": "E. Marinho Jr. (GitHub): маятник Уилберфорса, видео 30 кадров/с, угол поворота θ (рад)",
        "url": f"{_MAR}/theta_data.csv",
        "file": "wilberforce_theta_data.csv",
        "sha256": "7f6cc0ebca8be6b10147dfd451c57e4fe08315e9c7cda00e80b67eccae3855a2",
    },
})
WILBERFORCE_PHONE_MASS = 0.3989        # кг, масса телефона (Gallot et al.)
WILBERFORCE_RHO = 0.033                # м, датчик ускорения от оси пружины (Gallot et al.)
ELASTIC_MASSES = (400, 500, 700, 900, 1000)        # г
ELASTIC_K_STATIC = 30.2970062                      # Н/м, автор измерил грузами (constants.txt)
ELASTIC_L0_STATIC = 0.075                          # м, длина пружины без груза (constants.txt)

LOCAL_DATA = Path(__file__).resolve().parent / "data"     # запасная копия, если сайт не отдаёт файл


def download(source: str, cache_dir: str | Path | None = None) -> bytes:
    """Скачивает файл источника по HTTP и проверяет контрольную сумму SHA-256.

    Порядок: (1) кеш на диске; (2) скачивание по ссылке; (3) если скачать не удалось
    (например, сайт журнала требует вход) -- локальная копия в папке data/ рядом с кодом.
    """
    src = SOURCES[source]
    cache = Path(cache_dir or os.environ.get("OSC_LAB_CACHE", Path.home() / ".cache" / "osc_lab"))
    path = cache / src["file"]
    if not path.exists():
        try:
            req = urllib.request.Request(src["url"], headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                raw = resp.read()
        except Exception as err:                          # сеть, 403, требование входа
            local = LOCAL_DATA / src["file"]
            if not local.exists():
                raise RuntimeError(f"не удалось скачать {src['url']} ({err}); положите файл "
                                   f"{src['file']} в папку {LOCAL_DATA}") from err
            warnings.warn(f"{source}: скачать по ссылке не удалось ({type(err).__name__}), "
                          f"взята локальная копия data/{src['file']}")
            raw = local.read_bytes()
        if hashlib.sha256(raw).hexdigest() != src["sha256"]:
            raise ValueError(f"{source}: файл не совпадает с зафиксированной версией")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != src["sha256"]:
        raise ValueError(f"{source}: файл в кеше повреждён, удалите {path}")
    return raw


def _science_table(name: str, cache_dir=None) -> np.ndarray:
    with zipfile.ZipFile(io.BytesIO(download("science", cache_dir))) as z:
        return np.loadtxt(io.StringIO(z.read(name).decode("ascii")), comments="%")


# --- система 1: пружина ---------------------------------------------------------------
def load_airtrack(cache_dir=None):
    """Пружина, источник 1 (Science 2009): t (с) и положение тележки x (N, 1).

    Файл real_linear_h_1.txt: «номер опыта, время, x, v». Тележка едет по прямой дорожке,
    поэтому координата одна. Единицы x авторы не указали; на частоту это не влияет.
    """
    a = _science_table("real_linear_h_1.txt", cache_dir)
    return a[:, 1] - a[0, 1], a[:, 2:3]


def load_airtrack_derivatives(cache_dir=None):
    """Тот же опыт, real_linear_a_1.txt: x, скорость v, ускорение a, посчитанные авторами."""
    a = _science_table("real_linear_a_1.txt", cache_dir)
    return a[:, 2], a[:, 4], a[:, 3]                       # столбцы: опыт, t, x, a, v


# --- система 2: маятник ---------------------------------------------------------------
def load_science_pendulum(cache_dir=None):
    """Маятник, источник 1 (Science 2009): t и угол θ (рад), файл real_pend_h_1.txt.

    В записи есть пропуски (трекер терял груз) -- это не мешает: решение ОДУ считаем
    в тех моментах, где данные есть.
    """
    a = _science_table("real_pend_h_1.txt", cache_dir)
    return a[:, 1] - a[0, 1], a[:, 2:3]


def load_science_pendulum_derivatives(cache_dir=None):
    """Тот же опыт, real_pend_a_1.txt: θ, угловая скорость ω и ускорение α от авторов."""
    a = _science_table("real_pend_a_1.txt", cache_dir)
    return a[:, 2], a[:, 4], a[:, 3]                       # столбцы: опыт, t, θ, α, ω


def load_video_pendulum(cache_dir=None):
    """Маятник, источник 2 (видео, Торонто): t и угол θ (рад) (N, 1).

    В файле координаты груза в пикселях. Точка подвеса не записана, поэтому находим её
    сами: груз движется по дуге окружности, центр окружности -- это подвес (МНК для
    окружности). Угол = направление от подвеса к грузу. Первые 3 кадра -- момент отпускания.
    """
    df = pd.read_csv(io.BytesIO(download("video", cache_dir)))
    t, X, Y = (df.iloc[:, i].to_numpy(float) for i in range(3))
    A = np.column_stack([X, Y, np.ones_like(X)])
    c, *_ = np.linalg.lstsq(A, X**2 + Y**2, rcond=None)
    xc, yc = c[0] / 2, c[1] / 2
    th = np.arctan2(X - xc, -(Y - yc))
    return t[3:] - t[3], th[3:, None]


def video_author_angle(cache_dir=None):
    """Угол (рад), который записала программа автора видео (4-й столбец файла)."""
    df = pd.read_csv(io.BytesIO(download("video", cache_dir)))
    return df.iloc[3:, 0].to_numpy(float) - df.iloc[3, 0], np.radians(df.iloc[3:, 3].to_numpy(float))


# --- система 3: упругий маятник -------------------------------------------------------
def load_elastic(mass_g: int, cache_dir=None):
    """Упругий маятник (Wawrzynek): t и положение груза r = (x, y, z) (N, 3), метры.

    Видео снято сбоку, поэтому движение в плоскости: y = 0, z -- вверх, начало -- подвес.
    Запись начинается с момента, когда груз отпустили.
    Tracker пишет время с округлением (раз в ~69 кадров шаг 0.035 с вместо 0.0334 с),
    а кадры видео на самом деле равномерные -> ставим равномерную сетку времени.
    """
    txt = download(f"elastic_{mass_g}g", cache_dir).decode("utf-8")
    df = pd.read_csv(io.StringIO(txt.split("\n", 1)[1]))              # 1-я строка: имя объекта
    t_raw = df["t"].to_numpy(float)
    t = np.arange(len(t_raw)) * (t_raw[-1] - t_raw[0]) / (len(t_raw) - 1)
    pos = np.column_stack([df["x"].to_numpy(float), np.zeros(len(df)), df["y"].to_numpy(float)])
    # в начале видео груз ещё держат рукой: отбрасываем кадры до отпускания
    # (первый кадр, где груз сдвинулся больше чем на 3 мм, минус один кадр)
    moved = np.linalg.norm(pos - pos[0], axis=1) > 0.003
    i0 = max(int(np.argmax(moved)) - 1, 0)
    return t[i0:] - t[i0], pos[i0:]


# --- система 4: маятник Уилберфорса ---------------------------------------------------
def load_wilberforce_phone(cache_dir=None, skip: float = 6.0) -> dict:
    """Уилберфорс, источник 4а (Gallot et al.): телефон -- это сам груз. Ось z телефона идёт
    вдоль пружины. Возвращает словарь: t (с), acc (N, 3) -- ускорение без тяжести по осям x, y, z
    телефона (м/с²), gyro (N, 3) -- скорость поворота вокруг осей x, y, z (рад/с).

    Обработка: участок свободных колебаний (после отпускания + skip с -- первые
    секунды телефон ещё дёргается от руки; без последних 3 с) и фильтр 0.15-2 Гц (убирает медленный дрейф датчиков и мелкую дрожь).
    """
    text = download("wilberforce_phone", cache_dir).decode("utf-8-sig")
    df = pd.read_csv(io.StringIO(text.split("\n", 1)[1]), sep=";")
    a = df.iloc[:, :10].to_numpy(float)
    t = a[:, 9] / 1000.0
    win = 30
    gz = a[:, 8]
    rms = np.sqrt(np.convolve((gz - np.median(gz)) ** 2, np.ones(win) / win, mode="same"))
    keep = (t >= t[np.argmax(rms > 0.25 * rms.max())] + skip) & (t <= t[-1] - 3.0)
    t = t[keep]
    sos = signal.butter(4, [0.15, 2.0], btype="bandpass", fs=1 / np.median(np.diff(t)), output="sos")
    acc = signal.sosfiltfilt(sos, a[keep, 3:6], axis=0)
    gyro = signal.sosfiltfilt(sos, a[keep, 6:9], axis=0)
    return {"t": t - t[0], "acc": acc, "gyro": gyro}


def load_wilberforce_video(cache_dir=None):
    """Уилберфорс, источник 4б (Marinho): t (с), высота груза z (м) и угол поворота θ (рад)."""
    z = np.loadtxt(io.StringIO(download("wilberforce_video_z", cache_dir).decode("utf-8-sig")), delimiter=",")
    th = np.loadtxt(io.StringIO(download("wilberforce_video_theta", cache_dir).decode("utf-8-sig")),
                    delimiter=",")
    n = min(len(z), len(th))
    return z[:n, 0], np.column_stack([z[:n, 1], th[:n, 1]])


# ======================================================================================
# 5. Векторы параметров с ожидаемыми значениями
# ======================================================================================
@dataclass
class Param:
    name: str
    unit: str
    expected: float
    source: str
    meaning: str


def _lsq_from_derivatives(q, v, a, use_sin):
    """Ожидание «по формуле статьи»: подставляем x, v, a авторов в уравнение, линейный МНК."""
    f = np.sin(q) if use_sin else q
    (c_q, c_v, _), *_ = np.linalg.lstsq(np.column_stack([f, v, np.ones_like(q)]), a, rcond=None)
    return float(np.sqrt(-c_q)), float(-c_v)


def params_airtrack(cache_dir=None) -> list[Param]:
    w, c = _lsq_from_derivatives(*load_airtrack_derivatives(cache_dir), use_sin=False)
    src = "авторы Science 2009: их x, v, a + МНК"
    return [Param("w", "рад/с", w, src, "частота пружины √(k/m)"),
            Param("zeta", "—", c / (2 * w), src, "затухание")]


def params_science_pendulum(cache_dir=None) -> list[Param]:
    w0, gam = _lsq_from_derivatives(*load_science_pendulum_derivatives(cache_dir), use_sin=True)
    src = "авторы Science 2009: их θ, ω, α + МНК"
    return [Param("w0", "рад/с", w0, src, "частота маятника √(g/L)"),
            Param("gamma", "1/с", gam, src, "трение")]


def params_video_pendulum(cache_dir=None) -> list[Param]:
    """Ожидание: период по углу, который записала программа автора (среднее между
    пересечениями нуля сверху вниз), w0 = 2π/T. Трение -- не известно (ставим 0)."""
    t, a = video_author_angle(cache_dir)
    a = a - a.mean()
    cross = t[np.where((a[:-1] > 0) & (a[1:] <= 0))[0]]
    T = float(np.mean(np.diff(cross)))
    return [Param("w0", "рад/с", 2 * np.pi / T, f"программа автора: T = {T:.3f} с",
                  "частота маятника √(g/L)"),
            Param("gamma", "1/с", 0.0, "не известно", "трение")]


def params_elastic(mass_g: int) -> list[Param]:
    m = mass_g / 1000
    src = "автор: k = 30.30 Н/м, l0 = 7.5 см (грузами, без движения)"
    return [Param("k", "Н/м", ELASTIC_K_STATIC, src, "жёсткость пружины"),
            Param("ws", "рад/с", np.sqrt(ELASTIC_K_STATIC / m), "√(k/m)", "частота пружины"),
            Param("l0", "м", ELASTIC_L0_STATIC, src, "длина пружины без груза"),
            Param("gamma", "1/с", 0.0, "не известно", "трение")]


def params_wilberforce_phone() -> list[Param]:
    """Ожидание для 4а -- из статьи авторов данных: частоты двух мод и затухание."""
    src = "Gallot et al. 2022"
    return [Param("f1", "Гц", 0.539, src, "быстрая мода"), Param("f2", "Гц", 0.407, src, "медленная мода"),
            Param("gamma", "1/с", 2 * 0.013 * 2 * np.pi * 0.473, src + ": ζ = 0.013 -> γ = 2ζω", "трение")]


def params_wilberforce_video() -> list[Param]:
    """Ожидание для 4б -- из подгонки автора видео: z(t) = A cos(Bt) cos(Ct), B = 0.5004, C = 4.752 рад/с.
    Это биения двух мод с частотами C + B и C - B."""
    src = "автор видео: C ± B"
    return [Param("f1", "Гц", (4.752 + 0.5004) / (2 * np.pi), src, "быстрая мода"),
            Param("f2", "Гц", (4.752 - 0.5004) / (2 * np.pi), src, "медленная мода"),
            Param("gamma", "1/с", 0.0, "не известно", "трение")]


def compare(params: list[Param], found: dict, stderr: dict | None = None) -> pd.DataFrame:
    """Таблица «ожидаемое -> найдено»."""
    stderr = stderr or {}
    rows = []
    for p in params:
        f = found[p.name]
        diff = 100 * (f / p.expected - 1) if p.expected else np.nan
        rows.append({"параметр": p.name, "смысл": p.meaning, "ед.": p.unit, "ожидаемое": p.expected,
                     "откуда": p.source, "найдено": f, "±σ": stderr.get(p.name, np.nan),
                     "отличие, %": diff})
    return pd.DataFrame(rows)


# ======================================================================================
# 6-8. Солвер, оптимизатор, ошибка
# ======================================================================================
SOLVER = dict(method="DOP853", rtol=1e-8, atol=1e-10)    # Рунге-Кутта 8-го порядка


def solve(rhs, t, state0, *params) -> np.ndarray:
    """Численное решение любого из уравнений: массив (len(t), число переменных состояния)."""
    t = np.asarray(t, float)
    sol = solve_ivp(rhs, (t[0], t[-1]), state0, t_eval=t, args=params, **SOLVER)
    if not sol.success:
        raise RuntimeError(sol.message)
    return sol.y.T


def main_frequency(t, y) -> float:
    """Частота самого высокого пика спектра, рад/с (стартовая точка для оптимизатора).

    Для записи с пропусками сначала переводим её на равномерную сетку (интерполяция).
    """
    t, y = np.asarray(t, float), np.asarray(y, float)
    dt = np.median(np.diff(t))
    tu = np.arange(t[0], t[-1], dt)
    yu = np.interp(tu, t, y)
    n = 16 * len(tu)
    spec = np.abs(np.fft.rfft(yu - yu.mean(), n))
    f = np.fft.rfftfreq(n, dt)
    return 2 * np.pi * f[np.argmax(spec[1:]) + 1]


def errors(y, y_model) -> dict:
    """MAE -- средний модуль промаха (в единицах данных), MSE -- средний квадрат промаха."""
    y, y_model = np.asarray(y, float), np.asarray(y_model, float)
    e = y_model - y
    return {"MAE": float(np.nanmean(np.abs(e))), "MSE": float(np.nanmean(e**2))}


def start_from(params: list[Param]) -> dict:
    """Стартовая точка оптимизатора = ожидаемые значения из вектора параметров (раздел 5)."""
    return {p.name: p.expected for p in params}


@dataclass
class Fit:
    params: dict
    stderr: dict
    model: np.ndarray       # предсказание модели в точках данных
    errors: dict
    nfev: int
    extra: dict = field(default_factory=dict)


def _stderr(res) -> np.ndarray:
    """Стандартные ошибки параметров из якобиана (как в curve_fit)."""
    J = res.jac.toarray() if sparse.issparse(res.jac) else res.jac
    dof = max(len(res.fun) - len(res.x), 1)
    cov = np.linalg.pinv(J.T @ J) * (res.fun @ res.fun) / dof
    return np.sqrt(np.clip(np.diag(cov), 0, None))


def fit_spring(t, Y, start: dict, damping: bool = True) -> Fit:
    """Пружина: подбор по всем осям сразу.

    Y -- данные (N, d). Общие для всех осей: w (и zeta, если damping=True).
    У каждой оси свои начальное положение, начальная скорость и смещение нуля датчика.
    Старт: w, zeta -- из вектора параметров (start), начальные условия -- из первых точек.
    """
    t = np.asarray(t, float)
    Y = np.asarray(Y, float).reshape(len(t), -1)
    d = Y.shape[1]
    nphys = 2 if damping else 1
    off0 = Y.mean(axis=0)
    v0 = (Y[1] - Y[0]) / (t[1] - t[0])
    x0 = np.concatenate([[start["w"]], [max(start.get("zeta", 0.0), 0.0)] if damping else [],
                         Y[0] - off0, v0, off0])

    def model(p):
        w, zeta = p[0], (p[1] if damping else 0.0)
        q = p[nphys:]
        return solve(rhs_spring, t, q[:2 * d], w, zeta)[:, :d] + q[2 * d:]

    lo = [0.1] + ([0.0] if damping else []) + [-np.inf] * (3 * d)
    hi = [100.0] + ([1.0] if damping else []) + [np.inf] * (3 * d)
    res = least_squares(lambda p: (model(p) - Y).ravel(), x0, bounds=(lo, hi), method="trf",
                        x_scale="jac")
    se = _stderr(res)
    names = ["w", "zeta"][:nphys]
    params = dict(zip(names, map(float, res.x[:nphys])))
    stderr = dict(zip(names, map(float, se[:nphys])))
    if not damping:
        params["zeta"], stderr["zeta"] = 0.0, 0.0
    fitted = model(res.x)
    return Fit(params, stderr, fitted, errors(Y, fitted), int(res.nfev))


def fit_pendulum(t, theta, start: dict, linear: bool = False) -> Fit:
    """Маятник: подбор w0, gamma, начального угла и скорости, смещения нуля угла.

    linear=True -- вариация «малые углы» (sin θ ≈ θ).
    Длина нити выводится из частоты: L = g / w0².
    """
    t = np.asarray(t, float)
    th = np.asarray(theta, float).ravel()
    v0 = (th[1] - th[0]) / (t[1] - t[0])
    x0 = [start["w0"], max(start.get("gamma", 0.0), 0.0), th[0] - th.mean(), v0, th.mean()]

    def model(p):
        return solve(rhs_pendulum, t, [p[2], p[3]], p[0], p[1], linear)[:, 0] + p[4]

    lo, hi = [0.1, 0.0, -np.pi, -50, -1], [100, 10, np.pi, 50, 1]
    res = least_squares(lambda p: model(p) - th, x0, bounds=(lo, hi), method="trf", x_scale="jac")
    se = _stderr(res)
    w0, gam = map(float, res.x[:2])
    params = {"w0": w0, "gamma": gam, "L": G / w0**2}
    stderr = {"w0": float(se[0]), "gamma": float(se[1]), "L": float(2 * G / w0**3 * se[0])}
    fitted = model(res.x)[:, None]
    return Fit(params, stderr, fitted, errors(th[:, None], fitted), int(res.nfev))


def fit_elastic(t, pos, mass_g: int, damping: bool = True, seg: float = 2.0,
                linear: bool = False) -> Fit:
    """Упругий маятник: подбор ws (-> k = m ws²), l0, gamma по координатам x и z.

    linear=True -- вариация «линейная» (качание и прыжки независимы, см. rhs_elastic).

    Почему по кускам. Упругий маятник очень чувствителен к начальным условиям: маленькая
    ошибка в старте за 20 с вырастает в большую (почти хаос). Поэтому запись режется на
    куски по `seg` секунд; у каждого куска свои начальные положение и скорость, а пружина
    (ws, l0, gamma) и сдвиг точки подвеса -- общие для всех кусков.
    """
    t = np.asarray(t, float)
    X, Z = pos[:, 0], pos[:, 2]
    dt = t[1] - t[0]
    k = max(1, int(round((len(t) - 1) * dt / seg)))                   # число кусков
    bounds = np.linspace(0, len(t) - 1, k + 1).round().astype(int)    # куски примерно по seg с
    starts = list(bounds[:-1])
    pieces = [slice(a, b + 1) for a, b in zip(bounds[:-1], bounds[1:])]
    vx = savgol_filter(X, 9, 3, deriv=1, delta=dt)
    vz = savgol_filter(Z, 9, 3, deriv=1, delta=dt)
    m = mass_g / 1000
    nphys = 5 if damping else 4                            # ws, l0, [gamma], сдвиг x, сдвиг z
    phys0 = [np.sqrt(ELASTIC_K_STATIC / m), ELASTIC_L0_STATIC] + ([0.05] if damping else []) + [0, 0]
    ic0 = np.concatenate([[X[s], Z[s], vx[s], vz[s]] for s in starts])

    def simulate(p):
        ws, l0 = p[0], p[1]
        gam = p[2] if damping else 0.0
        ox, oz = p[nphys - 2], p[nphys - 1]
        out = []
        for j, sl in enumerate(pieces):
            x, z, ux, uz = p[nphys + 4 * j: nphys + 4 * j + 4]
            y = solve(rhs_elastic, t[sl] - t[sl.start], [x, 0, z, ux, 0, uz], ws, l0, gam, G, linear)
            out.append(np.column_stack([y[:, 0] + ox, y[:, 2] + oz]))
        return out

    def resid(p):
        return np.concatenate([(mdl - np.column_stack([X[sl], Z[sl]])).ravel()
                               for mdl, sl in zip(simulate(p), pieces)])

    # каждый кусок зависит только от общих параметров и от своего старта -> разреженный якобиан
    sizes = [2 * (sl.stop - sl.start) for sl in pieces]
    Jsp = sparse.lil_matrix((sum(sizes), nphys + 4 * len(pieces)), dtype=int)
    Jsp[:, :nphys] = 1
    row = 0
    for j, sz in enumerate(sizes):
        Jsp[row:row + sz, nphys + 4 * j: nphys + 4 * j + 4] = 1
        row += sz
    lo = [0.5, -0.5] + ([0.0] if damping else []) + [-0.1, -0.1] + [-np.inf] * len(ic0)
    hi = [50, 0.5] + ([5.0] if damping else []) + [0.1, 0.1] + [np.inf] * len(ic0)
    res = least_squares(resid, np.concatenate([phys0, ic0]), bounds=(lo, hi), method="trf",
                        x_scale="jac", jac_sparsity=Jsp)
    se = _stderr(res)
    ws, l0 = map(float, res.x[:2])
    params = {"ws": ws, "k": m * ws**2, "l0": l0, "gamma": float(res.x[2]) if damping else 0.0}
    stderr = {"ws": float(se[0]), "k": float(2 * m * ws * se[0]), "l0": float(se[1]),
              "gamma": float(se[2]) if damping else 0.0}
    model = np.full((len(t), 2), np.nan)
    for mdl, sl in zip(simulate(res.x), pieces):
        model[sl] = mdl
    data = np.column_stack([X, Z])
    return Fit(params, stderr, model, errors(data, model), int(res.nfev),
               extra={"piece_starts": t[starts], "n_pieces": len(pieces)})


def solve_linear(A, t, state0) -> np.ndarray:
    """Точное решение линейной системы state' = A·state через собственные числа:
    state(t) = V·exp(Λt)·V⁻¹·state0. Совпадает с DOP853 (проверено тестом), но в сотни раз быстрее."""
    lam, Vm = np.linalg.eig(A)
    E = np.exp(np.outer(np.asarray(t, float) - t[0], lam))
    return np.real(np.einsum("ik,nk,kj,j->ni", Vm, E, np.linalg.inv(Vm), np.asarray(state0, float)))


def _linear_basis(A, t):
    """Все 4 решения из единичных стартов сразу: (N, 4, 4), [n, переменная, старт]."""
    lam, Vm = np.linalg.eig(A)
    E = np.exp(np.outer(np.asarray(t, float) - t[0], lam))
    return np.real(np.einsum("ik,nk,kj->nij", Vm, E, np.linalg.inv(Vm)))


def _two_peaks(t, y) -> tuple[float, float]:
    """Две самые высокие частоты спектра (рад/с) -- старт для подбора двух мод."""
    dt = np.median(np.diff(t))
    n = 16 * len(y)
    S = np.abs(np.fft.rfft(y - y.mean(), n))
    f = np.fft.rfftfreq(n, dt)
    idx = signal.find_peaks(S)[0]
    idx = np.sort(idx[np.argsort(S[idx])[::-1][:2]])
    return 2 * np.pi * f[idx[1]], 2 * np.pi * f[idx[0]]


def fit_wilberforce(t, Y, kind: str, start: dict, coupled: bool = True) -> Fit:
    """Маятник Уилберфорса: подбор wz, wθ, bz, bθ, gamma.

    kind = "phone": Y = (z'' -- ускорение вдоль пружины, θ' -- гироскоп вокруг пружины);
    kind = "video": Y = (z, θ).
    Уравнение линейное, поэтому решение точное (solve_linear), а начальные условия и «нули»
    датчиков находятся обычным МНК при каждой пробе. Оптимизатор (Левенберг-Марквардт) подбирает
    только физические параметры.
    Старт -- из вектора параметров (start: f1, f2, gamma): wz = wθ = середина между модами,
    bz·bθ = ((w1² - w2²)/2)² (тогда формула мод даёт ровно f1 и f2), а bz/bθ = (размах z / размах θ)²:
    в моде с wz ≈ wθ амплитуды z и θ относятся как √bz : √bθ.
    """
    t = np.asarray(t, float)
    Y = np.asarray(Y, float)
    w1, w2 = 2 * np.pi * start["f1"], 2 * np.pi * start["f2"]
    kap = (w1**2 - w2**2) / 2
    g0 = start.get("gamma", 0.0)
    sc = Y.std(axis=0)
    N = len(t)

    def project(p):
        A = wilberforce_matrix(*p)
        B = _linear_basis(A, t)
        if kind == "phone":
            o1, o2 = np.einsum("j,njk->nk", A[2], B), B[:, 3]
        else:
            o1, o2 = B[:, 0], B[:, 1]
        D = np.zeros((2 * N, 6))
        D[:N, :4], D[N:, :4] = o1 / sc[0], o2 / sc[1]
        D[:N, 4], D[N:, 5] = 1 / sc[0], 1 / sc[1]
        yv = np.concatenate([Y[:, 0] / sc[0], Y[:, 1] / sc[1]])
        c, *_ = np.linalg.lstsq(D, yv, rcond=None)
        return D @ c - yv, D @ c

    if coupled:
        wm = np.sqrt((w1**2 + w2**2) / 2)
        r = sc[0] / sc[1] / (wm if kind == "phone" else 1.0)     # размах z / размах θ
        best = least_squares(lambda p: project(p)[0], [wm, wm, kap * r, kap / r, g0], method="lm",
                             x_scale="jac")
        p = best.x
    else:
        best = least_squares(lambda p: project([p[0], p[1], 0, 0, p[2]])[0], [w1, w2, g0], method="lm",
                             x_scale="jac")
        p = np.array([best.x[0], best.x[1], 0, 0, best.x[2]])
    J = best.jac
    dof = max(len(best.fun) - len(best.x), 1)
    se_x = np.sqrt(np.clip(np.diag(np.linalg.pinv(J.T @ J) * (best.fun @ best.fun) / dof), 0, None))
    se = np.zeros(5)
    if coupled:
        se[:] = se_x
    else:
        se[[0, 1, 4]] = se_x
    _, M = project(p)
    model = np.column_stack([M[:N] * sc[0], M[N:] * sc[1]])
    wz, wt, bz, bt, gam = map(float, p)
    m1, m2 = wilberforce_modes(wz, wt, bz, bt) if coupled else (max(wz, wt), min(wz, wt))
    params = {"wz": wz, "wt": wt, "bz": bz, "bt": bt, "gamma": gam,
              "f1": m1 / (2 * np.pi), "f2": m2 / (2 * np.pi)}
    stderr = dict(zip(["wz", "wt", "bz", "bt", "gamma"], map(float, se)))
    if coupled:      # σ частот мод: перенос ошибок через производные формулы мод по параметрам
        cov = np.linalg.pinv(J.T @ J) * (best.fun @ best.fun) / dof
        for k, idx in (("f1", 0), ("f2", 1)):
            grad = np.zeros(5)
            for j in range(4):
                dp = np.array(p[:4], float)
                h = 1e-6 * max(abs(dp[j]), 1e-3)
                dp[j] += h
                grad[j] = (wilberforce_modes(*dp)[idx] - (m1, m2)[idx]) / h
            stderr[k] = float(np.sqrt(max(grad @ cov @ grad, 0.0)) / (2 * np.pi))
    errs = {}
    for i, name in enumerate(("z", "θ")):
        e = errors(Y[:, i], model[:, i])
        errs[f"MAE {name}"], errs[f"MSE {name}"] = e["MAE"], e["MSE"]
    return Fit(params, stderr, model, errs, int(best.nfev))
