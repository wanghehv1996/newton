## 📊 物理参数深度解析（基于VBD Solver实现）

### 🎯 1. **布料弹性参数（Triangle Elasticity）**

#### **tri_ke** (Lamé's first parameter μ - 剪切模量)
- **范围**: 1.0 - 1000.0，默认 100.0
- **物理意义**: 控制布料的**剪切刚度**（抗变形能力）
- **代码实现**（line 200-256）:
  ```python
  # StVK (St. Venant-Kirchhoff) 能量密度模型
  # ψ = μ * ||G||_F² + 0.5 * λ * (trace(G))²
  # 其中 G 是 Green 应变张量
  ```
  - 计算变形梯度 F = [x01, x02] * DmInv
  - Green应变: G = 0.5(F^T F - I)
  - 第一 Piola-Kirchhoff 应力: PK1 = 2μFG + λtrace(G)F
  
- **效果**:
  - ⬆️ 增大 → 布料更硬，抗拉伸/压缩能力强
  - ⬇️ 减小 → 布料更软，易变形

#### **tri_ka** (Lamé's second parameter λ - 体积模量)
- **范围**: 1.0 - 1000.0，默认 100.0
- **物理意义**: 控制布料的**体积保持能力**（抗面积变化）
- **作用**: 防止三角形面积剧烈改变
- **效果**:
  - ⬆️ 增大 → 三角形面积更稳定，抗压缩
  - ⬇️ 减小 → 允许更多的面积变化

#### **tri_kd** (Triangle Damping)
- **范围**: 1e-8 - 1e-4，默认 1.5e-6
- **物理意义**: 布料弹性的**阻尼系数**，消耗振动能量
- **代码实现**（line 297-349）:
  ```python
  # 约束：Cμ = ||G||_F (Green应变的Frobenius范数)
  # 时间导数：dCμ/dt
  # 阻尼力：-μ * damping * (dCμ/dt) * (dCμ/dx)
  # 阻尼Hessian：μ * damping * (1/dt) * (dCμ/dx) ⊗ (dCμ/dx)
  ```
  
- **效果**:
  - ⬆️ 增大 → 快速抑制振荡，布料显得"粘稠"
  - ⬇️ 减小 → 布料振动持续时间长，显得"弹性"

### 🔄 2. **弯曲参数（Bending）**

#### **bending_ke** (Bending Stiffness)
- **范围**: 1e-6 - 1e-2，默认 1e-4
- **物理意义**: 控制布料的**抗折叠能力**
- **代码实现**（line 395-451）:
  ```python
  # 基于二面角的弯曲模型
  # 计算两个相邻三角形法向量n1, n2之间的夹角θ
  # θ = atan2(sin_θ, cos_θ)
  # 能量导数：dE/dθ = k * (θ - θ_rest)
  # 实际刚度：k = stiffness * edge_rest_length
  ```
  
- **效果**:
  - ⬆️ 增大 → 布料抗折叠，像纸板
  - ⬇️ 减小 → 布料易弯曲，像丝绸

#### **bending_kd** (Bending Damping)
- **范围**: 1e-5 - 1e-1，默认 1e-3
- **物理意义**: 弯曲运动的**阻尼**
- **代码实现**（line 502-526）:
  ```python
  # 角速度：dθ/dt = Σ(dθ/dxi · dxi/dt) / dt
  # 阻尼力：-damping * k * (dθ/dt) * (dθ/dx)
  # 阻尼Hessian：damping * k * (1/dt) * (dθ/dx) ⊗ (dθ/dx)
  ```
  
- **效果**:
  - ⬆️ 增大 → 快速消除折叠振荡
  - ⬇️ 减小 → 布料折叠后会持续"摆动"

### 💥 3. **接触参数（Contact）**

#### **cloth_particle_radius** (粒子半径)
- **范围**: 0.001 - 0.02，默认 0.008
- **物理意义**: 布料粒子的**碰撞检测半径**
- **用途**: 
  - 参与软接触(soft contact)的渗透深度计算
  - 影响布料的"厚度"感

#### **cloth_body_contact_margin** (接触边界)
- **范围**: 0.001 - 0.05，默认 0.01
- **物理意义**: 布料与刚体接触的**激活距离**
- **效果**: 提前检测接触，防止深度穿透

#### **soft_contact_ke** (软接触刚度)
- **范围**: 100.0 - 10000.0，默认 1000.0
- **物理意义**: 布料与刚体接触时的**弹性系数**
- **代码实现**（line 567-571）:
  ```python
  penetration_depth = -(n · (particle_pos - body_pos) - particle_radius)
  if penetration_depth > 0:
      force_norm = penetration_depth * soft_contact_ke
      force = n * force_norm
      hessian = soft_contact_ke * (n ⊗ n)
  ```
  
- **效果**:
  - ⬆️ 增大 → 接触更"硬"，反弹快，穿透少
  - ⬇️ 减小 → 接触更"软"，布料可以"陷入"物体

#### **soft_contact_kd** (软接触阻尼)
- **范围**: 1e-5 - 1e-1，默认 5e-3
- **物理意义**: 接触时的**能量耗散**
- **代码实现**（line 577-580）:
  ```python
  dx = particle_pos - particle_prev_pos
  if dot(n, dx) < 0:  # 粒子正在靠近物体
      damping_hessian = (soft_contact_kd / dt) * body_contact_hessian
      damping_force = -damping_hessian * dx
  ```
  
- **效果**:
  - ⬆️ 增大 → 接触无弹跳，像撞到海绵
  - ⬇️ 减小 → 接触有弹性，像撞到橡胶

### 🔗 4. **自接触参数（Self-Contact）**

#### **self_contact_radius** (自接触半径)
- **范围**: 0.0001 - 0.01，默认 0.001
- **物理意义**: 布料自身粒子间的**碰撞检测半径**
- **代码实现**（line 626-645）:
  ```python
  penetration_depth = collision_radius - distance
  # C2连续性计算
  tau = collision_radius * 0.5
  if tau > distance > 1e-5:
      k2 = 0.5 * tau² * k
      dEdD = -k2 / distance  # 平滑接触力
      d2E_dDdD = k2 / distance²
  else:
      dEdD = -k * penetration_depth  # 线性接触力
  ```
  
- **重要**: 必须 < self_contact_margin，建议 margin = 1.5~2 × radius

#### **self_contact_margin** (自接触边界)
- **范围**: 0.0001 - 0.01，默认 0.002
- **物理意义**: BVH碰撞检测的**查询距离**
- **作用**: 提前发现潜在碰撞，防止穿模
- **效果**:
  - 太小 → 漏检碰撞，布料穿透自己
  - 太大 → 过度检测，性能下降

#### **self_contact_friction** (自接触摩擦)
- **范围**: 0.0 - 2.0，默认 1.0
- **物理意义**: 布料自身的**摩擦系数** μ
- **作用**: 
  - 0 = 完全光滑
  - 1 = 正常布料
  - >1 = 高摩擦，像毛毡

### ⏱️ 5. **时间参数**

#### **fps** (帧率)
- **范围**: 10 - 120，默认 60
- **影响**:
  - `sim_dt = (1.0 / fps) / sim_substeps`
  - 所有阻尼项都除以 `dt`
  - VBD迭代中的 Hessian 包含 `1/dt` 项

## 🎓 参数调优建议

### 🧪 稳定性优先
```python
tri_ke = 100.0      # 中等刚度
tri_ka = 100.0      # 中等体积保持
tri_kd = 1e-5       # 较强阻尼
bending_ke = 1e-4   # 适中弯曲
bending_kd = 1e-2   # 较强弯曲阻尼
```

### 🎨 真实感优先
```python
tri_ke = 500.0      # 较高刚度（棉布）
tri_kd = 1e-6       # 弱阻尼（自然振动）
bending_ke = 1e-5   # 低弯曲（柔软）
self_contact_friction = 0.8  # 真实摩擦
```

### ⚡ 性能优先
```python
fps = 30            # 降低时间步精度
tri_kd = 5e-5       # 增加阻尼快速稳定
self_contact_margin = 0.003  # 减小检测范围
```
