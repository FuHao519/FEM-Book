import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'cm'

# ---------- 数据计算 ----------
n_arr = np.array([2, 4, 8, 16, 32, 64, 128, 256], dtype=float)
h_arr = 1.0 / n_arr                      # h = 1/n
pi_n  = n_arr * np.sin(np.pi / n_arr)
e_n   = np.abs(np.pi - pi_n)

# 过滤误差达到机器精度的点
mask  = e_n > 1e-15
h_use = h_arr[mask]
e_use = e_n[mask]

log_h = np.log10(h_use)                   # 横坐标: log₁₀(h), 从小到大
log_e = np.log10(e_use)                   # 纵坐标: log₁₀(e_n)

# ---------- 理论参考线 ----------
# log₁₀(e) = log₁₀(π³/6) + 2·log₁₀(h)   斜率 = +2
intercept = np.log10(np.pi**3 / 6.0)
x_ref = np.linspace(log_h.min() - 0.2, log_h.max() + 0.2, 300)
y_ref = intercept + 2.0 * x_ref

# ---------- 绘图 ----------
fig, ax = plt.subplots(figsize=(7, 5))

ax.plot(log_h, log_e, 'o-',
        color='#1565C0', linewidth=1.8, markersize=7,
        markerfacecolor='white', markeredgewidth=1.8,
        markeredgecolor='#1565C0', zorder=3,
        label=r'Computed $\log_{10}|\pi - \pi_n|$')

ax.plot(x_ref, y_ref, '--',
        color='#C62828', linewidth=1.3, alpha=0.85, zorder=2,
        label=r'Theory, slope $= +2$')

ax.set_xlabel(r'$\log_{10}\, h\quad (h=1/n)$', fontsize=14)
ax.set_ylabel(r'$\log_{10}\, e_n$', fontsize=14)
ax.set_title(r'Log–log convergence: $e_n = |\pi - \pi_n|, \quad '
             r'\pi_n = n\,\sin(\pi/n)$',
             fontsize=13, pad=10)

ax.legend(fontsize=12, loc='lower right', framealpha=0.9)
ax.grid(True, which='major', linewidth=0.5, alpha=0.3)
ax.tick_params(labelsize=11)

plt.tight_layout()
plt.savefig('pi_loglog_plot.png', dpi=150, bbox_inches='tight')
plt.show()
