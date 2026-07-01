import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib  # 日本語フォント対応

# 軸のデータ作成
x1 = np.linspace(0, 110, 400)

# 新しい制約条件の直線
# x1 + 2*x2 <= 100 -> x2 = (100 - x1) / 2
y1 = (100 - x1) / 2
# x1 + 3*x2 <= 90  -> x2 = (90 - x1) / 3
y2 = (90 - x1) / 3

def plot_lp(z_val, filename, step_title):
    plt.figure(figsize=(7, 6))
    
    # 領域の塗りつぶし
    y_min = np.minimum(y1, y2)
    plt.fill_between(x1, 0, np.maximum(0, y_min), where=(x1>=0), color='lightblue', alpha=0.5, label='実行可能領域')
    
    # 制約条件の境界線
    plt.plot(x1, y1, 'r-', linewidth=1.5, label='集中力制約: $x_1 + 2x_2$ <= 100')
    plt.plot(x1, y2, 'g-', linewidth=1.5, label='体力制約: $x_1 + 3x_2$ <= 90')
    
    # 目的関数 z = 40x1 + 20x2 -> x2 = -2 * x1 + z/20
    y_z = -2.0 * x1 + z_val / 20
    plt.plot(x1, y_z, 'b--', linewidth=2.5, label=f'目的関数直線 (z = {z_val})')
    
    # 重要な頂点をプロット
    vertices = [(0, 0), (0, 30), (90, 0)]
    for v in vertices:
        if v == (90, 0) and z_val == 3600:
            plt.plot(v[0], v[1], 'ro', markersize=9, label='最適解 (90, 0)')
        else:
            plt.plot(v[0], v[1], 'ko', markersize=5)
            
    if z_val == 3600:
        plt.annotate('最適解\n(90, 0) -> z=3600', xy=(90, 0), xytext=(50, 20),
                     arrowprops=dict(facecolor='black', shrink=0.08, width=1, headwidth=6))
        
    plt.xlim(0, 110)
    plt.ylim(0, 45)
    plt.xlabel('練習時間 $x_1$ (時間)', fontsize=11)
    plt.ylabel('試合時間 $x_2$ (時間)', fontsize=11)
    plt.title(f'{step_title}\n(z = {z_val})', fontsize=12, fontweight='bold')
    plt.legend(loc='upper right', fontsize='medium')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()

# 4つのステップで画像を生成
steps = [
    (1200, 'ステップ 1: 目的関数 z が領域を通過し始める', 'lp_step_1.png'),
    (1800, 'ステップ 2: 直線が頂点 (0, 30) を通過する', 'lp_step_2.png'),
    (3000, 'ステップ 3: 直線がさらに右側へ移動する', 'lp_step_3.png'),
    (3600, 'ステップ 4: 領域の右端の頂点 (90, 0) で最適解となる', 'lp_step_4.png')
]

for z, title, fname in steps:
    plot_lp(z, fname, title)

print("日本語版のグラフ画像の生成が完了しました！")