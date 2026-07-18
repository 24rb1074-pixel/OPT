import numpy as np


# 課題1：二次関数に対する3手法の比較
def f(x):
    return x[0]**2 + 10 * x[1]**2


def gradient(x):
    return np.array([2 * x[0], 20 * x[1]], dtype=float)


def hessian(x):
    return np.array([[2.0, 0.0], [0.0, 20.0]])


def backtracking_line_search(x, d, grad, alpha=1.0, beta=0.5, c=1e-4):
    # Armijo条件を満たすまで刻み幅を縮小する
    while f(x + alpha * d) > f(x) + c * alpha * np.dot(grad, d):
        alpha *= beta
    return alpha


def gradient_descent(x_0=np.array([3.0, 2.0]), max_iter=100, eps=1e-6):
    x = np.array(x_0, dtype=float, copy=True)
    path = [x.copy()]
    for k in range(max_iter):
        grad = gradient(x)
        # 3手法で共通の停止条件
        if np.linalg.norm(grad) < eps:
            return x, f(x), k, np.array(path)
        d = -grad
        alpha = backtracking_line_search(x, d, grad)
        x = x + alpha * d
        path.append(x.copy())
    return x, f(x), max_iter, np.array(path)


def newton(x_0=np.array([3.0, 2.0]), max_iter=100, eps=1e-6):
    x = np.array(x_0, dtype=float, copy=True)
    path = [x.copy()]
    for k in range(max_iter):
        grad = gradient(x)
        if np.linalg.norm(grad) < eps:
            return x, f(x), k, np.array(path)
        # 逆行列を明示的に計算せず、連立方程式からNewton方向を求める
        d = np.linalg.solve(hessian(x), -grad)
        x = x + d
        path.append(x.copy())
    return x, f(x), max_iter, np.array(path)


def BFGS(x_0=np.array([3.0, 2.0]), max_iter=100, eps=1e-6):
    x = np.array(x_0, dtype=float, copy=True)
    B = np.eye(len(x))  # Hesse行列の初期近似
    path = [x.copy()]
    for k in range(max_iter):
        grad = gradient(x)
        if np.linalg.norm(grad) < eps:
            return x, f(x), k, np.array(path)
        d = np.linalg.solve(B, -grad)
        alpha = backtracking_line_search(x, d, grad)
        s_k = (alpha * d).reshape(-1, 1)
        x_new = x + s_k.ravel()
        y_k = (gradient(x_new) - grad).reshape(-1, 1)
        # s_kと勾配差y_kからHesse行列の近似を更新する
        sBs = (s_k.T @ B @ s_k).item()
        ys = (y_k.T @ s_k).item()
        if sBs > 1e-12 and ys > 1e-12:
            B = B - (B @ s_k @ s_k.T @ B) / sBs + (y_k @ y_k.T) / ys
        x = x_new
        path.append(x.copy())
    return x, f(x), max_iter, np.array(path)


# 課題2：Himmelblau関数に対する初期値依存性の確認
def hb(x):
    return (x[0]**2 + x[1] - 11)**2 + (x[0] + x[1]**2 - 7)**2


def hb_gradient(x):
    # 共通部分を一度だけ計算する
    a = x[0]**2 + x[1] - 11
    b = x[0] + x[1]**2 - 7
    return np.array([4 * x[0] * a + 2 * b, 2 * a + 4 * x[1] * b])


def hb_hessian(x):
    return np.array([
        [12 * x[0]**2 + 4 * x[1] - 42, 4 * (x[0] + x[1])],
        [4 * (x[0] + x[1]), 4 * x[0] + 12 * x[1]**2 - 26]
    ])


def hb_line_search(x, d, grad, alpha=1.0, beta=0.5, c=1e-4):
    # 3手法で使用するArmijo直線探索
    while hb(x + alpha * d) > hb(x) + c * alpha * np.dot(grad, d):
        alpha *= beta
    return alpha


def hb_gradient_descent(x_0, max_iter=100, tol=1e-6):
    x = np.array(x_0, dtype=float, copy=True)
    path = [x.copy()]
    for k in range(max_iter):
        grad = hb_gradient(x)
        if np.linalg.norm(grad) < tol:
            return x, hb(x), k, np.array(path)
        d = -grad
        alpha = hb_line_search(x, d, grad)
        x = x + alpha * d
        path.append(x.copy())
    return x, hb(x), max_iter, np.array(path)


def hb_newton(x_0, max_iter=100, tol=1e-6):
    x = np.array(x_0, dtype=float, copy=True)
    path = [x.copy()]
    for k in range(max_iter):
        grad = hb_gradient(x)
        if np.linalg.norm(grad) < tol:
            return x, hb(x), k, np.array(path)
        d = np.linalg.solve(hb_hessian(x), -grad)
        alpha = hb_line_search(x, d, grad)
        x = x + alpha * d
        path.append(x.copy())
    return x, hb(x), max_iter, np.array(path)


def hb_BFGS(x_0, max_iter=100, tol=1e-6):
    x = np.array(x_0, dtype=float, copy=True)
    B = np.eye(len(x))
    path = [x.copy()]
    for k in range(max_iter):
        grad = hb_gradient(x)
        if np.linalg.norm(grad) < tol:
            return x, hb(x), k, np.array(path)
        d = np.linalg.solve(B, -grad)
        alpha = hb_line_search(x, d, grad)
        s_k = (alpha * d).reshape(-1, 1)
        x_new = x + s_k.ravel()
        y_k = (hb_gradient(x_new) - grad).reshape(-1, 1)
        sBs = (s_k.T @ B @ s_k).item()
        ys = (y_k.T @ s_k).item()
        # 分母が十分に正の場合のみ更新し、数値的不安定性を避ける
        if sBs > 1e-12 and ys > 1e-12:
            B = B - (B @ s_k @ s_k.T @ B) / sBs + (y_k @ y_k.T) / ys
        x = x_new
        path.append(x.copy())
    return x, hb(x), max_iter, np.array(path)


# 課題3：有効制約集合を仮定したKKT条件の求解
def kkt_objective(x, center):
    return np.sum((x - center)**2)


def g(x):
    # g_i(x) <= 0 の順に、円・x1非負・x2非負の制約を返す
    return np.array([x[0]**2 + x[1]**2 - 1, -x[0], -x[1]])


def grad_g(x):
    return [np.array([2*x[0], 2*x[1]]),
            np.array([-1.0, 0.0]), np.array([0.0, -1.0])]


def solve_active_set(center, active_set, max_iter=100, tol=1e-10):
    center = np.array(center, dtype=float)
    active_set = list(active_set)
    m = len(active_set)
    x = center.copy()
    # 仮定した境界上に初期点を配置する
    if 0 in active_set:
        x = center / np.linalg.norm(center)
    if 1 in active_set:
        x[0] = 0.0
    if 2 in active_set:
        x[1] = 0.0
    if 0 in active_set and 1 in active_set:
        x = np.array([0.0, 1.0 if center[1] >= 0 else -1.0])
    if 0 in active_set and 2 in active_set:
        x = np.array([1.0 if center[0] >= 0 else -1.0, 0.0])

    z = np.r_[x, np.zeros(m)]
    for iteration in range(max_iter):
        x, mu = z[:2], z[2:]
        gradients = grad_g(x)
        # 停留条件と有効制約の等式をまとめた非線形方程式F=0
        stationarity = 2*(x - center)
        for j, i in enumerate(active_set):
            stationarity += mu[j]*gradients[i]
        F = np.r_[stationarity, g(x)[active_set]]
        if np.linalg.norm(F) < tol:
            mu_all = np.zeros(3)
            mu_all[active_set] = mu
            return x, mu_all, iteration

        # KKT方程式のJacobian（鞍点行列）を構成する
        H = 2*np.eye(2)
        if 0 in active_set:
            H += 2*mu[active_set.index(0)]*np.eye(2)
        G = np.column_stack([gradients[i] for i in active_set]) \
            if m else np.empty((2, 0))
        J = np.block([[H, G], [G.T, np.zeros((m, m))]])
        try:
            z += np.linalg.solve(J, -F)
        except np.linalg.LinAlgError:
            return None
    return None


def verify_kkt_checklist(center, active_set, result, tol=1e-7):
    if result is None:
        return False, ["NG", "NG", "NG", "NG", "NG"]
    x, mu, _ = result
    gx = g(x)
    # 主実行可能性、有効集合の整合性、双対実行可能性、相補性
    c1 = np.all(gx <= tol)
    c2 = all(abs(gx[i]) <= tol for i in active_set) if active_set else True
    c3 = all(abs(mu[i]) <= tol for i in range(3) if i not in active_set)
    c4 = all(mu[i] >= -tol for i in active_set) if active_set else True
    c5 = np.all(np.abs(mu * gx) <= tol)
    # チェックリストに加え、停留条件も最終判定に含める
    stationarity = 2*(x - np.array(center, dtype=float))
    for i in range(3):
        stationarity += mu[i]*grad_g(x)[i]
    c_stat = np.linalg.norm(stationarity) <= tol
    is_valid = c1 and c2 and c3 and c4 and c5 and c_stat
    checklist = ["OK" if cond else "NG" for cond in (c1, c2, c3, c4, c5)]
    return is_valid, checklist
