import numpy as np

def simplex(c, A, b, sense="max", verbose=False):
    """
    一般化されたシンプレックス法アルゴリズム (z行を1行目に配置)
    
    引数:
        c (array-like): 長さ n の目的関数の係数ベクトル
        A (array-like): サイズ (m x n) の制約式の係数行列
        b (array-like): 長さ m の制約の右辺ベクトル (b >= 0)
        sense (str): "max" または "min"
        verbose (bool): 途中経過（タブローの更新）を出力するかどうか
        
    戻り値:
        dict: 計算結果を含む辞書
    """
    # Numpy配列に変換（計算時の型エラーを防ぐためfloat型に）
    c = np.array(c, dtype=float)
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    
    m, n = A.shape # m: 制約式の数, n: 元の変数の数
    
    # ========================================================
    # 【初期化】タブローの構築
    # 行数: m + 1 (目的関数 1行 + 制約式 m行)
    # 列数: n + m + 1 (変数x n列 + スラックs m列 + RHS 1列)
    # ========================================================
    rows = m + 1
    cols = n + m + 1
    tableau = np.zeros((rows, cols))
    
    # 1. 目的関数のセット (z行を0行目に配置)
    for j in range(n):
        if sense == "max":
            tableau[0, j] = -c[j]
        else:
            tableau[0, j] = c[j]
            
    # 2. 制約式のセット (A と スラック変数 と b を 1行目以降に配置)
    for i in range(m):
        for j in range(n):
            tableau[i + 1, j] = A[i, j]
        tableau[i + 1, n + i] = 1.0  # スラック変数の列に1を立てる
        tableau[i + 1, -1] = b[i]    # RHS
        
    # 基底変数の添字リスト (0〜n-1 が x、n〜n+m-1 が s を表す)
    # 初期状態の基底はすべてスラック変数（1行目からm行目に対応）
    basis = [n + i for i in range(m)]

    # --- ヘルパー関数: 添字から変数名を生成 (表示用) ---
    def get_var_name(idx):
        if idx < n:
            return f"x{idx + 1}"
        else:
            return f"s{idx - n + 1}"

    # --- ヘルパー関数: タブローのフォーマット出力 ---
    def print_tableau(step_num):
        print(f"\n--- [Iteration {step_num}] ---")
        headers = ["基底"] + [f"x{j+1}" for j in range(n)] + [f"s{i+1}" for i in range(m)] + ["RHS"]
        header_str = " | ".join([f"{h:>5}" for h in headers])
        print(header_str)
        print("-" * len(header_str))
        for i in range(rows):
            # 0行目は "z"、1行目以降は対応する基底変数を表示
            b_name = "z" if i == 0 else get_var_name(basis[i - 1])
            row_str = f"{b_name:>5} | " + " | ".join([f"{tableau[i, j]:>5.2f}" for j in range(cols - 1)]) + f" | {tableau[i, -1]:>5.2f}"
            print(row_str)
        print()

    if verbose:
        print("【初期タブロー】")
        print_tableau(0)

    num_iterations = 0
    status = "optimal"
    
    # ========================================================
    # 【メインループ】シンプレックス法の反復
    # ========================================================
    while True:
        # ----------------------------------------------------
        # フェーズ1: 入る変数を選ぶ（Dantzig規則：0行目の最小値）
        # ----------------------------------------------------
        pivot_col = -1
        min_val = -1e-9 # 浮動小数点誤差を考慮した閾値
        
        for j in range(cols - 1): # RHSは調べない
            if tableau[0, j] < min_val:
                min_val = tableau[0, j]
                pivot_col = j
                
        # 停止条件: z行に負の係数がない (被約費用がすべて >= 0)
        if pivot_col == -1:
            break
            
        if verbose:
            print(f"➔ フェーズ1: {get_var_name(pivot_col)} が入る変数 (Dantzig規則)")

        # ----------------------------------------------------
        # フェーズ2: 出る変数を選ぶ（最小比検定）
        # ----------------------------------------------------
        pivot_row = -1
        min_ratio = float('inf')
        
        # z行(0行目)は除外し、1行目から調べる
        for i in range(1, rows):
            element = tableau[i, pivot_col]
            if element > 1e-9: # 係数が正の場合のみ比を計算
                ratio = tableau[i, -1] / element
                if ratio < min_ratio:
                    min_ratio = ratio
                    pivot_row = i
                    
        # 停止条件(非有界判定): すべての係数が 0 以下
        if pivot_row == -1:
            status = "unbounded"
            if verbose:
                print("➔ 終了: 最小比を計算できる正の係数がないため、解は非有界 (Unbounded) です。")
            break
            
        if verbose:
            # pivot_rowは1以降なので、basisのインデックスはpivot_row - 1
            print(f"➔ フェーズ2: {get_var_name(basis[pivot_row - 1])} が出る変数 (最小比 = {min_ratio:.2f})")

        # ----------------------------------------------------
        # フェーズ3: ピボット操作（行基本変形）の明示実装
        # ----------------------------------------------------
        pivot_element = tableau[pivot_row, pivot_col]
        
        if verbose:
            print(f"➔ フェーズ3: ピボット要素 {pivot_element:.2f} で行基本変形を実行")

        # (A) ピボット行全体をピボット要素で割り、交差点を 1 にする
        for j in range(cols):
            tableau[pivot_row, j] /= pivot_element
            
        # (B) 他の行からピボット行を引いて、ピボット列を 0 にする（掃き出し）
        for i in range(rows):
            if i != pivot_row:
                factor = tableau[i, pivot_col]
                for j in range(cols):
                    tableau[i, j] -= factor * tableau[pivot_row, j]
                    
        # 基底変数の添字を更新
        basis[pivot_row - 1] = pivot_col
        num_iterations += 1
        
        if verbose:
            print_tableau(num_iterations)

    # ========================================================
    # 【結果の抽出】
    # ========================================================
    # 主の最適解 x
    x = np.zeros(n)
    if status == "optimal":
        for i in range(m):
            if basis[i] < n: # スラック変数ではなく、元の変数 x の場合
                # basis[i]はtableauの i+1 行目に対応している
                x[basis[i]] = tableau[i + 1, -1]
                
    # 最適値 (0行目のRHS)
    objective = tableau[0, -1]
    if sense == "min":
        objective = -objective
        
    # 双対最適解 y* (最終タブローの0行目のスラック変数部分から読み取る)
    y = np.zeros(m)
    if status == "optimal":
        for i in range(m):
            y[i] = tableau[0, n + i]

    return {
        "status": status,
        "x": x.tolist(),
        "objective": objective,
        "y": y.tolist(),
        "num_iterations": num_iterations,
        "basis": basis
    }


# ============================================================
# テスト実行用コード (演習問題3のデータを使用)
# ============================================================
if __name__ == "__main__":
    c = [48,35,27,52,31,40,23,45]
    A = [[3,2,1,4,2,3,1,2],
        [2,3,2,2,1,2,3,4],
        [1,1,2,2,1,1,1,2],
        [4,1,0,5,2,3,0,2],
        [0,2,3,1,2,1,4,3],
        [2,2,1,3,1,2,1,2]]
    b = [240,220,150,200,180,160]
    
    print("=== シンプレックス法の実行開始 ===")
    result = simplex(c, A, b, sense="max", verbose=True)
    
    print("=== 最終結果 (戻り値 dict の内容) ===")
    for key, value in result.items():
        if isinstance(value, float):
            print(f"{key:15}: {value:.2f}")
        else:
            print(f"{key:15}: {value}")