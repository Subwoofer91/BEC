import numpy as np
import matplotlib.pyplot as plt

class GPE2D_Solver:
    """二维 Gross-Pitaevskii 方程求解器"""
    def __init__(self, N=256, L=20.0, dt=0.01):
        self.N = N
        self.L = L
        self.dx = L / N
        self.dt = dt
        
        # 预分配实空间网格
        self.x = np.linspace(-L/2, L/2, N, endpoint=False)
        self.y = np.linspace(-L/2, L/2, N, endpoint=False)
        self.X, self.Y = np.meshgrid(self.x, self.y)
        
        # 预分配动量空间网格与演化算符
        kx = np.fft.fftfreq(N, d=self.dx) * 2 * np.pi
        ky = np.fft.fftfreq(N, d=self.dx) * 2 * np.pi
        KX, KY = np.meshgrid(kx, ky)
        
        self.T_kin = 0.5 * (KX**2 + KY**2)
        self.exp_T = np.exp(-self.T_kin * dt)
        
        # 波函数初始化
        self.psi = np.exp(-(self.X**2 + self.Y**2) / 2.0)
        self.normalize()
        
    def normalize(self):
        """归一化并进行发散检查"""
        psi_norm = np.sqrt(np.sum(np.abs(self.psi)**2) * self.dx**2)
        if np.isnan(psi_norm) or psi_norm == 0:
            raise ValueError("数值不稳定：波函数发散或坍缩！请检查物理参数或时间步长。")
        self.psi /= psi_norm
        
    def evolve(self, steps, g, gamma_QF, use_trap=True, use_LHY=True):
        """虚时演化核心循环"""
        V_trap = 0.5 * (self.X**2 + self.Y**2) if use_trap else np.zeros_like(self.X)
        
        for step in range(steps):
            # 计算局部非线性势能
            V_total = V_trap + g * np.abs(self.psi)**2
            if use_LHY:
                V_total += gamma_QF * np.abs(self.psi)**3 # 加入 LHY 修正 [cite: 58]
                
            # Strang splitting 演化
            self.psi *= np.exp(-V_total * (self.dt / 2.0))
            
            psi_k = np.fft.fft2(self.psi)
            psi_k *= self.exp_T
            self.psi = np.fft.ifft2(psi_k)
            
            V_total = V_trap + g * np.abs(self.psi)**2
            if use_LHY:
                V_total += gamma_QF * np.abs(self.psi)**3
                
            self.psi *= np.exp(-V_total * (self.dt / 2.0))
            
            # 每步结束进行归一化
            self.normalize()
            
            # 在实际科研中，通常会在能量差低于例如 10^-7 时判定为收敛 [cite: 119]
            # 这里为了保持简洁，固定步数演化

# --- 主程序入口 ---
if __name__ == "__main__":
    # 参数设置
    g_val = -50.0         # 吸引接触相互作用
    gamma_val = 100.0     # LHY 排斥修正
    trap_flag = False     # 关闭势阱以观察自束缚液滴
    lhy_flag = True       
    
    # 实例化求解器并运行
    solver = GPE2D_Solver(N=256, L=20.0, dt=0.01)
    try:
        solver.evolve(steps=2000, g=g_val, gamma_QF=gamma_val, 
                      use_trap=trap_flag, use_LHY=lhy_flag)
    except ValueError as e:
        print(f"演化终止: {e}")
        exit()

    # --- 结果可视化 ---
    density = np.abs(solver.psi)**2
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # 1. 2D 密度伪彩图
    c1 = ax1.pcolormesh(solver.X, solver.Y, density, shading='auto', cmap='viridis')
    fig.colorbar(c1, ax=ax1, label='Density $|\psi|^2$')
    ax1.set_title(f"2D Density\n(g={g_val}, Trap={'ON' if trap_flag else 'OFF'}, LHY={'ON' if lhy_flag else 'OFF'})")
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.axis('equal')
    
    # 2. 1D 截面图 (y=0)
    mid_idx = solver.N // 2
    ax2.plot(solver.x, density[mid_idx, :], 'b-', lw=2)
    ax2.set_title("1D Cross-section (y=0)")
    ax2.set_xlabel('x')
    ax2.set_ylabel('Density')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()