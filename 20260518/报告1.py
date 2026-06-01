# 导入数值计算库numpy，用于数组、矩阵、矢量运算
import numpy as np

def truss3d_element_stiffness(x1, x2, E, A):
    """
    计算三维杆单元的单元刚度矩阵（全局坐标系）
    参数:
        x1: 节点1坐标 [x1, y1, z1]
        x2: 节点2坐标 [x2, y2, z2]
        E: 弹性模量 (Pa)
        A: 横截面积 (m^2)
    返回:
        L: 单元长度 (m)
        direction_cosines: 方向余弦 [cx, cy, cz]
        Ke: 6x6 单元刚度矩阵 (N/m)
    """
    # 将坐标列表转为浮点型numpy数组，方便矢量运算
    x1 = np.asarray(x1, dtype=float)
    x2 = np.asarray(x2, dtype=float)

    # 计算两节点坐标差值矢量 (Δx, Δy, Δz)
    dx = x2 - x1
    # 计算杆件实际长度（矢量模长）
    L = np.linalg.norm(dx)
    # 判断单元是否退化：两节点几乎重合，长度趋近于0则报错
    if L < 1e-12:
        raise ValueError("错误：两个节点重合，单元退化！")

    # 计算方向余弦：杆件轴线与全局x/y/z轴夹角的余弦值
    cx, cy, cz = dx / L

    # 杆件轴向拉压刚度系数 k = EA/L (材料力学基本公式)
    k = E * A / L
    # 把方向余弦存入一维数组
    c = np.array([cx, cy, cz])
    # 外积运算，生成3×3方向余弦乘积矩阵 c*c^T
    ccT = np.outer(c, c)

    # 按照三维桁架单元理论，拼接得到6×6整体坐标系单元刚度矩阵
    Ke = k * np.block([[ccT, -ccT],
                       [-ccT, ccT]])

    # 返回杆长、方向余弦、单元刚度矩阵
    return L, (cx, cy, cz), Ke


def truss3d_element_stress(x1, x2, E, A, de):
    """
    根据单元节点位移计算应变、应力和轴力
    参数:
        x1, x2, E, A: 同上
        de: 节点位移列阵 [u1, v1, w1, u2, v2, w2] (m)
    返回:
        epsilon: 轴向应变 (无量纲)
        sigma: 轴向应力 (Pa)
        N: 轴力 (N，受拉为正)
    """
    # 坐标转为浮点数组
    x1 = np.asarray(x1, dtype=float)
    x2 = np.asarray(x2, dtype=float)
    # 计算节点坐标差矢量
    dx = x2 - x1
    # 计算杆件长度
    L = np.linalg.norm(dx)
    # 长度过小判定，防止计算异常
    if L < 1e-12:
        raise ValueError("错误：两个节点重合，无法计算应力！")
    # 求解方向余弦
    cx, cy, cz = dx / L
    # 位移向量转为浮点数组
    de = np.asarray(de, dtype=float)

    # 解包位移向量：节点1三向位移、节点2三向位移
    u1, v1, w1, u2, v2, w2 = de

    # 相对位移向杆轴投影，得到杆件轴向总伸长量 ΔL
    delta = (u2 - u1) * cx + (v2 - v1) * cy + (w2 - w1) * cz

    # 轴向应变 ε = ΔL / L
    epsilon = delta / L
    # 胡克定律求应力 σ = E·ε
    sigma = E * epsilon
    # 轴力 N = σ·A （拉力为正，压力为负）
    N = sigma * A

    # 返回应变、应力、轴力
    return epsilon, sigma, N


# 程序入口：验证算例，仅当前文件直接运行时执行
if __name__ == "__main__":
    # 设置numpy输出格式：保留4位小数、取消科学计数法
    np.set_printoptions(precision=4, suppress=True)

    # 分割线 + 标题：一维轴向杆算例
    print("=" * 60)
    print("算例1：沿x轴的一维杆单元")
    print("=" * 60)
    # 定义节点1坐标 (原点)
    x1 = [0, 0, 0]
    # 定义节点2坐标 (沿x轴2m处)
    x2 = [2, 0, 0]
    # 弹性模量 200GPa
    E = 200e9
    # 杆件横截面积 1e-4 m²
    A = 1.0e-4
    # 节点位移：节点1不动，节点2沿x向位移1mm
    de = [0, 0, 0, 1.0e-3, 0, 0]

    # 调用函数计算杆长、方向余弦、单元刚度矩阵
    L, (cx, cy, cz), Ke = truss3d_element_stiffness(x1, x2, E, A)
    # 打印单元长度
    print(f"单元长度 L = {L:.4f} m")
    # 打印三个方向余弦
    print(f"方向余弦 (cx, cy, cz) = ({cx:.0f}, {cy:.0f}, {cz:.0f})")
    # 换行输出刚度矩阵标题
    print("\n刚度矩阵 Ke (6x6):")
    # 打印6阶单元刚度矩阵
    print(Ke)

    # 调用函数计算应变、应力、轴力
    epsilon, sigma, N = truss3d_element_stress(x1, x2, E, A, de)
    # 输出轴向应变
    print(f"\n轴向应变 epsilon = {epsilon:.4e}")
    # 输出应力（转换为MPa单位）
    print(f"轴向应力 sigma = {sigma / 1e6:.1f} MPa")
    # 输出杆件轴力
    print(f"轴力 N = {N:.2e} N")

    # 分割线 + 标题：空间斜杆算例
    print("\n" + "=" * 60)
    print("算例2：空间任意方向杆单元")
    print("=" * 60)
    # 节点1坐标原点
    x1 = [0, 0, 0]
    # 空间斜向节点2坐标 (1,2,2)
    x2 = [1, 2, 2]
    # 弹性模量 210GPa
    E = 210e9
    # 横截面积 2e-4 m²
    A = 2.0e-4
    # 节点2三向位移均为1/2/2 mm
    de = [0, 0, 0, 1.0e-3, 2.0e-3, 2.0e-3]

    # 计算杆长、方向余弦、刚度矩阵
    L, (cx, cy, cz), Ke = truss3d_element_stiffness(x1, x2, E, A)
    # 打印杆长
    print(f"单元长度 L = {L:.4f} m")
    # 打印空间杆方向余弦
    print(f"方向余弦 (cx, cy, cz) = ({cx:.4f}, {cy:.4f}, {cz:.4f})")
    # 输出刚度矩阵标题并打印矩阵
    print("\n刚度矩阵 Ke (6x6):")
    print(Ke)

    # 校验刚度矩阵是否对称（有限元基本性质）
    print(f"\n刚度矩阵是否对称? {np.allclose(Ke, Ke.T)}")

    # 计算刚度矩阵所有特征值，判断半正定性
    eigvals = np.linalg.eigvalsh(Ke)
    print(f"刚度矩阵特征值: {eigvals}")
    # 提取最小特征值
    min_eig = np.min(eigvals)
    # 判断特征值是否非负（允许极小浮点误差）
    if min_eig >= -1e-10:
        print("特征值均非负（半正定），满足要求。")
    else:
        print(f"警告：存在负特征值 {min_eig:.2e}，计算可能有问题。")
    # 说明：自由单元存在刚体位移，刚度矩阵奇异，必有零特征值
    print("说明：单个自由杆单元包含刚体位移模式，因此刚度矩阵奇异（存在零特征值）。")

    # 构造刚体平移位移：两节点位移完全相同，杆件无拉伸压缩
    de_rigid = [0.1, 0.2, 0.3, 0.1, 0.2, 0.3]
    # 位移乘以刚度矩阵，得到节点力向量
    Fe_rigid = Ke @ de_rigid
    print(f"\n刚体平移位移产生的节点力（应全为零）: {Fe_rigid}")
    # 校验刚体平移是否不产生内力
    if np.allclose(Fe_rigid, 0, atol=1e-10):
        print("刚体平移不会产生内力，验证通过。")
    else:
        print("警告：刚体平移产生了非零节点力。")

    # 计算该空间杆在给定位移下的应变、应力、轴力
    epsilon, sigma, N = truss3d_element_stress(x1, x2, E, A, de)
    print(f"\n轴向应变 epsilon = {epsilon:.4e}")
    print(f"轴向应力 sigma = {sigma / 1e6:.1f} MPa")
    print(f"轴力 N = {N:.2e} N")

    # 分割线 + 标题：刚度矩阵物理意义验证
    print("\n" + "=" * 60)
    print("任务4：刚度矩阵物理意义验证")
    print("=" * 60)
    # 选定自由度序号：索引3 对应节点1的z向位移 w1
    j = 3
    # 初始化6维位移向量，全部置0
    de_unit = np.zeros(6)
    # 令第j个自由度产生**单位位移(1)**，其余固定
    de_unit[j] = 1.0
    # 单位位移下，求解对应的节点力向量
    Fe = Ke @ de_unit
    print(f"令第{j + 1}个自由度（w1）位移为1，其他为0时，单元节点力列阵 Fe =")
    print(Fe)
    # 说明节点力向量等价于刚度矩阵对应列
    print("\n该节点力列阵即为刚度矩阵的第{}列（索引{}）".format(j + 1, j))
    print("Ke 的第{}列:".format(j + 1))
    # 取出刚度矩阵第j列并打印
    print(Ke[:, j])
    # 解释刚度系数 k_ij 的物理含义（有限元核心概念）
    print("\n物理意义：k_ij 表示当第 j 个自由度产生单位位移时，为了保持平衡需要在第 i 个自由度上施加的节点力。")
    print("本例中，当节点1的z向位移为1时，杆件产生轴向伸长，需要在两个节点上施加沿杆轴方向的力以平衡。")