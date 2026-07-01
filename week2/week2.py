import numpy as np

def init_tableau():
    coef = np.array([[1, 1],
                     [1, 3]])
    c_T = np.array([-2, -3])
    RHS = np.array([4, 6])
    
    # 3行5列の純粋な数値配列
    tableau = np.zeros((3, 5))
    
    for i in range(2):
        tableau[i, 0] = coef[i, 0]
        tableau[i, 1] = coef[i, 1]
        tableau[i, 2 + i] = 1.0   
        tableau[i, 4] = RHS[i]     
        
    tableau[2, 0] = c_T[0]
    tableau[2, 1] = c_T[1]
    
    # 【追加】行ごとの基底変数の名前をリストで定義
    # 0行目はs1、1行目はs2、2行目はzを意味する
    basis_labels = ["s1", "s2", "z"]
    
    return tableau, basis_labels

def show_tableau(tableau, basis):
    # ヘッダーの一番左に「基底」を追加
    print("基底  |   x1   x2   s1   s2  |  RHS")
    print("------------------------------------")
    for i in range(tableau.shape[0]):
        # ループごとに、リストから対応する基底変数の文字列（basis[i]）を取り出して左端に出力
        print(f" {basis[i]:<4} | {tableau[i, 0]:>4.1f} {tableau[i, 1]:>4.1f} {tableau[i, 2]:>4.1f} {tableau[i, 3]:>4.1f} | {tableau[i, 4]:>5.1f}")
    print()

# 実行
tab, basis = init_tableau()

import numpy as np

def simplex_step_explicit(tableau, basis):
    rows = tableau.shape[0] # 行数 (3)
    cols = tableau.shape[1] # 列数 (5)

    # ========================================================
    # 1. 列を選ぶ（一番利益の上がる変数 = z行の最小値を探す）
    # ========================================================
    pivot_col = 0
    min_val = 0.0
    
    # 最後の行（rows - 1）をチェック。RHS（最後の列）は調べないので cols - 1 まで。
    for j in range(cols - 1):
        if tableau[rows - 1, j] < min_val:
            min_val = tableau[rows - 1, j]
            pivot_col = j

    print(f"➔ 【フェーズ1】 {pivot_col+1}列目(x{pivot_col+1}) を増やします")


    # ========================================================
    # 2. 行を選ぶ（限界値の計算 = 最小比テスト）
    # ========================================================
    pivot_row = -1
    min_ratio = float('inf') # 最初は無限大を入れておく

    # 目的関数の行は調べないので rows - 1 まで。
    for i in range(rows - 1):
        element = tableau[i, pivot_col]
        # 係数がプラスの場合のみ、どれだけ増やせるか計算できる
        if element > 0:
            ratio = tableau[i, cols - 1] / element # RHS ÷ 係数
            if ratio < min_ratio:
                min_ratio = ratio
                pivot_row = i

    print(f"➔ 【フェーズ2】 {pivot_row}行目(現在の基底: {basis[pivot_row]}) が限界なので入れ替えます")


    # ========================================================
    # 3. ピボット操作（行基本変形）
    # ========================================================
    pivot_element = tableau[pivot_row, pivot_col]

    # (A) まず、ピボット行全体をピボット要素で割って、交差点を「1」にする
    for j in range(cols):
        tableau[pivot_row, j] /= pivot_element

    # (B) 他の行について、ピボット列が「0」になるように引き算する
    for i in range(rows):
        if i != pivot_row: # ピボット行以外を処理
            factor = tableau[i, pivot_col] # 消したい値
            for j in range(cols):
                tableau[i, j] -= (factor * tableau[pivot_row, j])

    # ========================================================
    # 4. 基底変数のラベルを更新
    # ========================================================
    basis[pivot_row] = f"x{pivot_col + 1}"
    
    print("➔ 【フェーズ3】 更新後のタブロー:")
    show_tableau(tableau, basis)
    return tableau, basis

def simplex_method(tableau, basis):
    while np.any(tableau[-1, :-1] < 0):  # 最終行の係数に負の値がある限り繰り返す
        tableau, basis = simplex_step_explicit(tableau, basis)
    return tableau, basis

final_tableau, final_basis = simplex_method(tab, basis)