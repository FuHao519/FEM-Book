"""
truss_solver.py - 杆/桁架结构有限元求解器
支持一维杆单元和二维桁架单元
使用缩减法处理位移边界条件
"""

import numpy as np
import json
import sys
import os


# 全局模型数据
class Model:
    pass


model = Model()


# 前处理
def read_json(filename):
    """读取JSON文件，填充model数据"""
    with open(filename, 'r') as f:
        data = json.load(f)

    model.title = data.get("Title", "Truss Analysis")
    model.nsd = data["nsd"]  # 空间维度 (1 or 2)
    model.ndof = data["ndof"]  # 每个节点自由度数 (通常等于nsd)
    model.nnp = data["nnp"]  # 节点总数
    model.nel = data["nel"]  # 单元总数
    model.nen = data["nen"]  # 每个单元节点数 (2)

    # 材料与几何
    model.E = np.array(data["E"])  # 弹性模量 (nel,)
    model.CArea = np.array(data["CArea"])  # 横截面积 (nel,)
    model.x = np.array(data["x"])  # 节点x坐标 (nnp,)
    model.y = np.array(data["y"]) if model.nsd >= 2 else np.zeros(model.nnp)

    # 单元连接关系 (IEN) - JSON中用1-based节点编号
    IEN_raw = data["IEN"]  # list of [node1, node2]
    model.IEN = np.array(IEN_raw, dtype=int) - 1  # 转为0-based索引

    # 边界条件 (自由度编号为1-based)
    fixed_dof = np.array(data["fixed_dof"]) - 1  # 转为0-based自由度号
    fixed_value = np.array(data["fixed_value"])
    model.fixed_dof = fixed_dof
    model.fixed_value = fixed_value

    force_dof = np.array(data["force_dof"]) - 1
    force_value = np.array(data["force_value"])
    model.force_dof = force_dof
    model.force_value = force_value

    # 计算总自由度数
    model.neq = model.nnp * model.ndof

    # 初始化总体刚度矩阵和载荷向量
    model.K = np.zeros((model.neq, model.neq))
    model.f = np.zeros(model.neq)

    # 施加节点载荷
    for dof, val in zip(force_dof, force_value):
        model.f[dof] = val

    # 预分配LM矩阵 (nen*ndof, nel)
    model.LM = np.zeros((model.nen * model.ndof, model.nel), dtype=int)


def set_LM():
    """建立对号矩阵LM: 单元局部自由度 -> 全局自由度号"""
    ndof = model.ndof
    for e in range(model.nel):
        # 获取该单元的全局节点号 (0-based)
        nodes = model.IEN[e]  # [n1, n2]
        for j in range(model.nen):  # 局部节点编号 0,1
            for m in range(ndof):  # 局部自由度编号 0..ndof-1
                ind = j * ndof + m  # 单元内自由度行/列索引
                global_dof = nodes[j] * ndof + m
                model.LM[ind, e] = global_dof


# 单元分析
def element_length(e):
    """计算单元长度和方向余弦 (c, s)"""
    nodes = model.IEN[e]
    x1, y1 = model.x[nodes[0]], model.y[nodes[0]]
    x2, y2 = model.x[nodes[1]], model.y[nodes[1]]
    dx = x2 - x1
    dy = y2 - y1
    L = np.sqrt(dx * dx + dy * dy)
    if model.ndof == 1:
        c, s = 1.0, 0.0  # 一维问题，无方向余弦概念
    else:
        c = dx / L
        s = dy / L
    return L, c, s


def truss_element_ke(e):
    """计算单元刚度矩阵 (nen*ndof) x (nen*ndof)"""
    L, c, s = element_length(e)
    EA = model.E[e] * model.CArea[e]
    const = EA / L

    if model.ndof == 1:
        ke = const * np.array([[1, -1],
                               [-1, 1]])
    elif model.ndof == 2:
        c2 = c * c
        s2 = s * s
        cs = c * s
        ke = const * np.array([[c2, cs, -c2, -cs],
                               [cs, s2, -cs, -s2],
                               [-c2, -cs, c2, cs],
                               [-cs, -s2, cs, s2]])
    else:
        raise ValueError("Only ndof=1 or 2 are supported in basic version")
    return ke


#  组装
def assembly():
    """将单元刚度矩阵组装到总体刚度矩阵"""
    for e in range(model.nel):
        ke = truss_element_ke(e)
        lm_e = model.LM[:, e]  # 单元对应的全局自由度列表
        # 直接组装
        for i in range(len(lm_e)):
            for j in range(len(lm_e)):
                model.K[lm_e[i], lm_e[j]] += ke[i, j]
        # 也可使用 numpy 的 ix_ 进行向量化组装（可选）
        # idx = np.ix_(lm_e, lm_e)
        # model.K[idx] += ke


#  边界条件处理与求解
def solve_system():
    """
    缩减法求解总体刚度方程
    将自由度分为已知位移(E)和未知位移(F)
    """
    neq = model.neq
    # 已知位移自由度（固定自由度）
    fixed = model.fixed_dof
    # 未知位移自由度 = 所有自由度中去掉已知的
    free = np.setdiff1d(np.arange(neq), fixed)

    # 分块矩阵
    K_FF = model.K[np.ix_(free, free)]
    K_EF = model.K[np.ix_(fixed, free)]
    f_F = model.f[free]
    d_E = model.fixed_value  # 已知位移值

    # 求解未知位移
    rhs = f_F - K_EF.T @ d_E
    d_F = np.linalg.solve(K_FF, rhs)

    # 重构总体位移向量
    d = np.zeros(neq)
    d[fixed] = d_E
    d[free] = d_F
    model.d = d

    # 计算约束反力: r = K @ d - f
    r = model.K @ d - model.f
    # 只输出已知自由度上的反力（即约束反力）
    model.r = r[fixed]

    # 打印位移和反力
    print("\n=== 节点位移 ===")
    for node in range(model.nnp):
        start = node * model.ndof
        if model.ndof == 1:
            print(f"节点 {node + 1}: 位移 = {d[start]:.6f}")
        else:
            u, v = d[start], d[start + 1]
            print(f"节点 {node + 1}: u = {u:.6f}, v = {v:.6f}")

    print("\n=== 约束反力 ===")
    for dof, val in zip(model.fixed_dof, model.r):
        node = dof // model.ndof
        local_dof = dof % model.ndof
        dir_str = "x" if local_dof == 0 else "y"
        print(f"自由度 {dof + 1} (节点{node + 1},{dir_str}): 反力 = {val:.6f}")


#  后处理
def postprocess():
    """计算并输出每个单元的应力与轴力"""
    print("\n=== 单元结果 ===")
    for e in range(model.nel):
        L, c, s = element_length(e)
        # 提取单元节点位移
        nodes = model.IEN[e]
        de = []
        for j in range(model.nen):
            node = nodes[j]
            start = node * model.ndof
            for m in range(model.ndof):
                de.append(model.d[start + m])
        de = np.array(de)

        # 计算应力
        EA = model.E[e] * model.CArea[e]
        if model.ndof == 1:
            # sigma = E/L * [-1, 1] * de
            B = np.array([-1, 1]) / L
            sigma = model.E[e] * (B @ de)
        else:
            # sigma = E/L * [-c, -s, c, s] * de
            B = np.array([-c, -s, c, s]) / L
            sigma = model.E[e] * (B @ de)

        axial_force = sigma * model.CArea[e]

        print(f"\n单元 {e + 1}:")
        print(f"  长度 = {L:.6f}")
        if model.ndof == 2:
            print(f"  方向余弦: c = {c:.6f}, s = {s:.6f}")
        print(f"  应力 = {sigma:.6f}")
        print(f"  轴力 = {axial_force:.6f}")


#  辅助检查
def check_K_properties():
    """检查总体刚度矩阵的对称性、奇异性等"""
    print("\n=== 总体刚度矩阵性质检查 ===")
    # 对称性
    diff = np.max(np.abs(model.K - model.K.T))
    print(f"最大不对称量: {diff:.2e} (应为0或极小值)")

    # 奇异性 (施加边界条件前)
    eigvals = np.linalg.eigvals(model.K)
    zero_eigs = np.sum(np.abs(eigvals) < 1e-10)
    print(f"零特征值个数(奇异度): {zero_eigs} (至少应为 {model.ndof})")

    # 对角元非负
    diag = np.diag(model.K)
    neg_diag = np.sum(diag < -1e-10)
    print(f"负对角元个数: {neg_diag}")

    print("\n总体刚度矩阵 (前几行几列):")
    print(model.K[:min(6, model.neq), :min(6, model.neq)])


# 主程序
def main(input_file):
    # 1. 前处理
    read_json(input_file)
    set_LM()

    # 2. 组装
    assembly()

    # 3. 输出总体刚度矩阵性质（可选）
    check_K_properties()

    # 4. 求解
    solve_system()

    # 5. 后处理
    postprocess()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("用法: python truss_solver.py <input.json>")
        sys.exit(1)
    input_file = sys.argv[1]
    if not os.path.exists(input_file):
        print(f"错误: 文件 {input_file} 不存在")
        sys.exit(1)
    main(input_file)