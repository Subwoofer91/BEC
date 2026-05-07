"""
液滴自束缚性三合一诊断脚本
=========================================
目标：严格验证 LHY.py 中找到的"疑似液滴"是否真的是自束缚液滴

三个测试：
  A. 初态无关性：不同宽度初态应收敛到同一 (RMS, E)
  B. 盒子无关性：不同盒子大小应给出同一结果（排除边界效应）
  C. 粒子数扫描：寻找自束缚临界 N_c，验证液滴标度律

注：这里沿用 LHY.py 中的准-2D 玩具模型（LHY 幂次 |ψ|^3），
    归一化约定从 ∫|ψ|²=1 改为 ∫|ψ|²=N，这样才能做测试 C。
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass


# =========================================================
# 1. 求解器（从 LHY.py 改造，加入 N-归一化 + 能量诊断）
# =========================================================

@dataclass
class Result:
    """单次演化的诊断结果"""
    psi: np.ndarray
    N: float           # 总粒子数
    E_kin: float       # 动能
    E_int: float       # 接触相互作用能
    E_LHY: float       # LHY 修正能
    E_tot: float       # 总能
    rms: float         # 密度的 RMS 宽度 √⟨r²⟩
    converged: bool    # 是否收敛（能量变化 < tol）


class DropletSolver:
    """带 N-归一化和能量诊断的 2D eGPE 求解器（玩具模型）"""

    def __init__(self, N_grid=256, L=20.0, dt=0.01):
        self.N_grid = N_grid
        self.L = L
        self.dx = L / N_grid
        self.dt = dt

        # 实空间网格
        self.x = np.linspace(-L/2, L/2, N_grid, endpoint=False)
        self.y = np.linspace(-L/2, L/2, N_grid, endpoint=False)
        self.X, self.Y = np.meshgrid(self.x, self.y)
        self.R2 = self.X**2 + self.Y**2

        # 动量空间
        kx = np.fft.fftfreq(N_grid, d=self.dx) * 2 * np.pi
        ky = np.fft.fftfreq(N_grid, d=self.dx) * 2 * np.pi
        self.KX, self.KY = np.meshgrid(kx, ky)
        self.T_kin = 0.5 * (self.KX**2 + self.KY**2)
        self.exp_T = np.exp(-self.T_kin * dt)

    def init_gaussian(self, width, N_particles):
        """初始化高斯波包，归一化到 N 粒子"""
        psi = np.exp(-self.R2 / (2 * width**2))
        norm = np.sqrt(np.sum(np.abs(psi)**2) * self.dx**2)
        psi *= np.sqrt(N_particles) / norm
        return psi

    def normalize_to_N(self, psi, N_particles):
        """将波函数归一化到固定粒子数 N"""
        norm = np.sqrt(np.sum(np.abs(psi)**2) * self.dx**2)
        if np.isnan(norm) or norm == 0:
            raise ValueError("数值发散")
        return psi * np.sqrt(N_particles) / norm

    def compute_energy(self, psi, g, gamma, use_LHY=True):
        """计算能量泛函的三个分量（注意：这里的 ψ 已经归一化到 N）"""
        density = np.abs(psi)**2

        # 动能：用动量空间计算更准确
        psi_k = np.fft.fft2(psi) * self.dx**2
        E_kin = np.sum(self.T_kin * np.abs(psi_k)**2) / (self.L**2)

        # 相互作用和 LHY
        E_int = 0.5 * g * np.sum(density**2) * self.dx**2
        E_LHY = (2.0/5.0) * gamma * np.sum(density**2.5) * self.dx**2 if use_LHY else 0.0

        return E_kin, E_int, E_LHY

    def compute_rms(self, psi):
        """计算密度的 RMS 宽度 √⟨r²⟩/N"""
        density = np.abs(psi)**2
        N_particles = np.sum(density) * self.dx**2
        return np.sqrt(np.sum(self.R2 * density) * self.dx**2 / N_particles)

    def evolve(self, psi_init, N_particles, g, gamma,
               max_steps=10000, tol=1e-7, check_every=200,
               use_trap=False, use_LHY=True):
        """虚时演化到收敛，返回完整的 Result"""
        psi = psi_init.copy().astype(complex)
        V_trap = 0.5 * self.R2 if use_trap else 0.0

        E_prev = None
        converged = False

        for step in range(max_steps):
            # Strang splitting: 半步势能 → 整步动能 → 半步势能
            density = np.abs(psi)**2
            V_loc = V_trap + g * density
            if use_LHY:
                V_loc += gamma * density**1.5   # δE/δψ* 对 (2/5)γ|ψ|^5 变分 = γ|ψ|^3·ψ

            psi *= np.exp(-V_loc * self.dt / 2.0)
            psi = np.fft.ifft2(np.fft.fft2(psi) * self.exp_T)

            density = np.abs(psi)**2
            V_loc = V_trap + g * density
            if use_LHY:
                V_loc += gamma * density**1.5
            psi *= np.exp(-V_loc * self.dt / 2.0)

            psi = self.normalize_to_N(psi, N_particles)

            # 收敛检查
            if step % check_every == 0 and step > 0:
                E_kin, E_int, E_LHY = self.compute_energy(psi, g, gamma, use_LHY)
                E_now = E_kin + E_int + E_LHY
                if E_prev is not None and abs(E_now - E_prev) < tol * abs(E_now + 1e-12):
                    converged = True
                    break
                E_prev = E_now

        E_kin, E_int, E_LHY = self.compute_energy(psi, g, gamma, use_LHY)
        return Result(
            psi=psi, N=N_particles,
            E_kin=E_kin, E_int=E_int, E_LHY=E_LHY,
            E_tot=E_kin + E_int + E_LHY,
            rms=self.compute_rms(psi),
            converged=converged,
        )


# =========================================================
# 2. 三个测试
# =========================================================

# 物理参数（和你 LHY.py 一致，方便对照）
G_VAL = -50.0
GAMMA_VAL = 100.0
N_PARTICLES = 1.0   # 默认粒子数；测试 C 会扫描它


def test_A_initial_state_independence():
    """测试 A：不同初始宽度 → 应收敛到相同的液滴"""
    print("\n" + "="*60)
    print("测试 A：初态无关性")
    print("="*60)
    solver = DropletSolver(N_grid=256, L=20.0, dt=0.01)
    widths = [0.5, 1.0, 2.0, 3.0]
    results = []
    for w in widths:
        psi0 = solver.init_gaussian(width=w, N_particles=N_PARTICLES)
        res = solver.evolve(psi0, N_PARTICLES, G_VAL, GAMMA_VAL)
        results.append((w, res))
        conv = "✓" if res.converged else "✗"
        print(f"  初始 σ={w:.1f} | RMS={res.rms:.4f} | E={res.E_tot:.4f} "
              f"| (E_kin,E_int,E_LHY)=({res.E_kin:.3f},{res.E_int:.3f},{res.E_LHY:.3f}) "
              f"| 收敛 {conv}")

    rms_vals = [r.rms for _, r in results]
    E_vals = [r.E_tot for _, r in results]
    rms_spread = (max(rms_vals) - min(rms_vals)) / np.mean(rms_vals)
    E_spread = (max(E_vals) - min(E_vals)) / abs(np.mean(E_vals))
    print(f"\n  RMS 相对分散 = {rms_spread*100:.2f}%")
    print(f"  E   相对分散 = {E_spread*100:.2f}%")
    print(f"  判据：若均 < 1%，则初态无关 → 真液滴；否则要警惕。")
    return results


def test_B_box_size_independence():
    """测试 B：不同盒子大小 → 排除边界效应"""
    print("\n" + "="*60)
    print("测试 B：盒子无关性")
    print("="*60)
    configs = [(20.0, 256), (30.0, 384), (40.0, 512)]
    results = []
    for L, N_grid in configs:
        solver = DropletSolver(N_grid=N_grid, L=L, dt=0.01)
        psi0 = solver.init_gaussian(width=1.0, N_particles=N_PARTICLES)
        res = solver.evolve(psi0, N_PARTICLES, G_VAL, GAMMA_VAL)
        results.append((L, N_grid, res))
        conv = "✓" if res.converged else "✗"
        print(f"  L={L}, N_grid={N_grid}, dx={L/N_grid:.4f} | "
              f"RMS={res.rms:.4f} | E={res.E_tot:.4f} | 收敛 {conv}")

    rms_vals = [r.rms for *_, r in results]
    rms_spread = (max(rms_vals) - min(rms_vals)) / np.mean(rms_vals)
    print(f"\n  RMS 相对分散 = {rms_spread*100:.2f}%")
    print(f"  判据：< 1% 表明波包在盒子中心自束缚，与边界无关。")
    return results


def test_C_particle_number_scan():
    """测试 C：扫描粒子数 N → 寻找自束缚临界 N_c"""
    print("\n" + "="*60)
    print("测试 C：粒子数扫描（寻找液滴临界 N_c）")
    print("="*60)
    solver = DropletSolver(N_grid=256, L=30.0, dt=0.01)
    N_list = [0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0]
    results = []
    for N in N_list:
        psi0 = solver.init_gaussian(width=1.5, N_particles=N)
        res = solver.evolve(psi0, N, G_VAL, GAMMA_VAL, max_steps=6000)
        results.append((N, res))
        # 判断是否"看起来像液滴"：能量为负 + 没有摊到盒子边
        is_bound = (res.E_tot < 0) and (res.rms < solver.L / 4)
        tag = "束缚" if is_bound else "扩散"
        print(f"  N={N:5.2f} | RMS={res.rms:6.3f} | E/N={res.E_tot/N:+.4f} "
              f"| E={res.E_tot:+.4f} [{tag}]")
    print("\n  判据：存在 N_c，使得 N>N_c 时 E<0 且 RMS 有限；")
    print("        N<N_c 时波包扩散（RMS 接近盒子尺度）。")
    return results


# =========================================================
# 3. 可视化
# =========================================================

def plot_all(results_A, results_B, results_C):
    fig = plt.figure(figsize=(15, 10))

    # --- 测试 A：初态无关性 ---
    ax1 = plt.subplot(2, 3, 1)
    widths = [w for w, _ in results_A]
    rms_A = [r.rms for _, r in results_A]
    ax1.plot(widths, rms_A, 'o-', lw=2, ms=8)
    ax1.set_xlabel('Initial gaussian width σ')
    ax1.set_ylabel('Converged RMS')
    ax1.set_title('Test A: RMS vs initial width\n(flat line → initial-state independent)')
    ax1.grid(True)

    ax2 = plt.subplot(2, 3, 2)
    # 画不同初态收敛后的 1D 剖面
    solver = DropletSolver(N_grid=256, L=20.0, dt=0.01)
    for w, r in results_A:
        density_1d = np.abs(r.psi[solver.N_grid // 2, :])**2
        ax2.plot(solver.x, density_1d, lw=2, label=f'σ_init={w}')
    ax2.set_xlabel('x')
    ax2.set_ylabel('|ψ|²')
    ax2.set_title('Test A: converged density profiles\n(should overlap)')
    ax2.legend(fontsize=8)
    ax2.grid(True)
    ax2.set_xlim(-5, 5)

    # --- 测试 B：盒子无关性 ---
    ax3 = plt.subplot(2, 3, 3)
    Ls = [L for L, _, _ in results_B]
    rms_B = [r.rms for _, _, r in results_B]
    E_B = [r.E_tot for _, _, r in results_B]
    ax3.plot(Ls, rms_B, 'o-', lw=2, ms=8, label='RMS')
    ax3_twin = ax3.twinx()
    ax3_twin.plot(Ls, E_B, 's--', color='red', lw=2, ms=8, label='E_tot')
    ax3.set_xlabel('Box size L')
    ax3.set_ylabel('RMS', color='C0')
    ax3_twin.set_ylabel('E_tot', color='red')
    ax3.set_title('Test B: vs box size\n(both flat → no boundary effect)')
    ax3.grid(True)

    # --- 测试 C：粒子数扫描 ---
    ax4 = plt.subplot(2, 3, 4)
    Ns = [N for N, _ in results_C]
    rms_C = [r.rms for _, r in results_C]
    ax4.semilogx(Ns, rms_C, 'o-', lw=2, ms=8)
    ax4.set_xlabel('Particle number N')
    ax4.set_ylabel('RMS width')
    ax4.set_title('Test C: RMS vs N\n(RMS should saturate when bound)')
    ax4.grid(True, which='both')

    ax5 = plt.subplot(2, 3, 5)
    E_C = [r.E_tot for _, r in results_C]
    E_per_N = [r.E_tot / N for N, r in results_C]
    ax5.semilogx(Ns, E_per_N, 'o-', lw=2, ms=8, color='green')
    ax5.axhline(0, color='k', ls=':', alpha=0.5)
    ax5.set_xlabel('Particle number N')
    ax5.set_ylabel('E_tot / N  (chemical potential ~)')
    ax5.set_title('Test C: energy per particle\n(E/N<0 → bound)')
    ax5.grid(True, which='both')

    # --- 最终液滴的 2D 密度 ---
    ax6 = plt.subplot(2, 3, 6)
    best = results_A[1][1]  # 取初态 σ=1.0 的结果
    density_2d = np.abs(best.psi)**2
    im = ax6.pcolormesh(solver.X, solver.Y, density_2d,
                        shading='auto', cmap='viridis')
    plt.colorbar(im, ax=ax6, label='|ψ|²')
    ax6.set_title(f'Converged droplet (N={N_PARTICLES})\n'
                  f'E={best.E_tot:.3f}, RMS={best.rms:.3f}')
    ax6.set_xlabel('x')
    ax6.set_ylabel('y')
    ax6.set_aspect('equal')

    plt.tight_layout()
    plt.savefig('/home/claude/droplet_diagnostics.png', dpi=120, bbox_inches='tight')
    plt.show()
    print("\n图像已保存到 droplet_diagnostics.png")


# =========================================================
# 4. 主程序
# =========================================================

if __name__ == "__main__":
    print("液滴自束缚性三合一诊断")
    print(f"参数: g = {G_VAL}, γ = {GAMMA_VAL}, N (默认) = {N_PARTICLES}")

    results_A = test_A_initial_state_independence()
    results_B = test_B_box_size_independence()
    results_C = test_C_particle_number_scan()

    plot_all(results_A, results_B, results_C)

    print("\n" + "="*60)
    print("诊断完成。最终判据汇总：")
    print("="*60)
    print("  真液滴应同时满足：")
    print("    [A] 不同初态收敛到同一 (RMS, E)")
    print("    [B] 不同盒子大小结果一致")
    print("    [C] 存在明确的临界 N_c，N>N_c 时 E<0 且 RMS 饱和")
    print("  任何一个不满足，都要回到参数或代码去查原因。")
