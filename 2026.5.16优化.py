import numpy as np
import pandas as pd

from mpl_toolkits.mplot3d import Axes3D
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.config import Config
import joblib
from matplotlib import pyplot as plt
from pylab import mpl
# 设置全局字体为新罗马字体

plt.rcParams['font.sans-serif'] = ['Times New Roman']  # 使用新罗马字体
# 禁用pymoo未编译警告
Config.warnings['not_compiled'] = False
plt.rcParams['axes.unicode_minus'] = False
# ------------------------------
# 1. 加载训练好的模型
# ------------------------------
model1 = joblib.load('2026.5.16.model_blow_loss.pkl')   # 目标函数2预测模型
model2 = joblib.load('2026.5.16.model_carbon.pkl')      # 约束因变量预测模型

# ------------------------------
# 2. 固定生产前批次信息
# ------------------------------
xd0 = np.array([0.029, 1308, 1423, 83, 247])
xd = xd0.reshape(1, -1)

# ------------------------------
# 3. 定义优化问题
# ------------------------------
class MyProblem(ElementwiseProblem):
    def __init__(self):
        n_var = 10
        xl = np.array([10496, 56, 0, 4748, 120, 0, 0, 0, 0, 23.7])#寻优空间下限
        xu = np.array([14606, 5974, 1378, 12115, 7713, 1120, 1500, 1891, 4224, 57.15])#寻优空间上限
        super().__init__(n_var=n_var, n_obj=2, n_constr=0, xl=xl, xu=xu)

    def _evaluate(self, x, out, *args, **kwargs):
        x_int = np.round(x).reshape(1, -1)

        yangqi   = x_int[0, 0] * 477 / 840
        N        = x_int[0, 1] * 554 / 800
        Ar       = x_int[0, 2] * 750 / 560
        baihui   = x_int[0, 3] * 600 / 1000
        shengbaiyunshi = x_int[0, 4] * 180 / 1000
        guitie   = x_int[0, 5] * 6300 / 1000
        guimeng  = x_int[0, 6] * 6200 / 1000
        tuoyangji= x_int[0, 7] * 19000 / 1000
        mengtie  = x_int[0, 8] * 42400 / 1000
        feigang  = x_int[0, 9] * 2200

        x_full = np.concatenate((xd, x_int), axis=1)
        sunshi = model1.predict(x_full)
        c = model2.predict(x_full)

        u, l = 0.1, 0.021
        penalty = 0.0
        penalty1 = 0.0
        if c > u:
            penalty = (c - u) * 10000000
            penalty1 = (c - u) * 100
        elif c < l:
            penalty = (l - c) * 10000000
            penalty1 = (l - c) * 100

        obj1 = yangqi + N + Ar + baihui + shengbaiyunshi + guitie + guimeng + tuoyangji + mengtie + feigang + penalty#目标函数1，penalty为软约束的惩罚项
        obj2 = sunshi + penalty1#目标函数2被惩罚

        out["F"] = [float(obj1), float(obj2)]

# ------------------------------
# 4. 运行NSGA2优化
# ------------------------------
problem = MyProblem()
algorithm = NSGA2(
    pop_size=50,
    crossover=SBX(prob=0.9, eta=15),
    mutation=PM(prob=0.1, eta=20),
    eliminate_duplicates=True
)

res = minimize(problem,
               algorithm,
               ('n_gen', 200),
               seed=42,
               sampling=FloatRandomSampling(),
               verbose=False)

print(f"优化完成，非支配解数量: {len(res.X)}")

# 保存结果到Excel
variables_cols = [f'Var_{i+1}' for i in range(res.X.shape[1])]
df_result = pd.DataFrame(res.X, columns=variables_cols)
df_result['obj1'] = res.F[:, 0]
df_result['obj2'] = res.F[:, 1]
df_result.to_excel('optimization_results.xlsx', index=False)
print("Pareto解集已保存至 optimization_results.xlsx")

# ------------------------------
# 5. 可视化（完全复用原代码风格）
# ------------------------------

# ---- 2D帕累托前沿图 ----
plt.figure(figsize=(8, 6))
# 使用原代码中的一种颜色（原来有两个颜色，这里取第一个颜色 #1874CD 或自定义）
plt.scatter(res.F[:, 0], res.F[:, 1], color=(187/255, 196/255, 255/255), s=25, label="NSGA2")
plt.title("Pareto solution", fontsize=20)
plt.xlabel("obj1", fontsize=20)
plt.ylabel("obj2", fontsize=20)
plt.legend()
plt.grid(False)
plt.tight_layout()
plt.savefig('pareto_front_2d.png', dpi=300)
plt.show()

# ---- 3D散点图（含碳含量约束平面） ----
# 计算所有解对应的碳含量
all_carbons = []
for x in res.X:
    x_int = np.round(x).reshape(1, -1)
    x_full = np.concatenate((xd, x_int), axis=1)
    c_pred = model2.predict(x_full)[0]
    all_carbons.append(c_pred)
all_carbons = np.array(all_carbons)

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# 颜色映射
colors_3d = (240/255, 23/255, 23/255)
ax.scatter(res.F[:, 0], res.F[:, 1], all_carbons,
           color=colors_3d, marker='o', label="NSGA2")

# 添加约束平面
carbon_levels = [0.021, 0.1]
cost_min, cost_max = res.F[:, 0].min(), res.F[:, 0].max()
blow_min, blow_max = res.F[:, 1].min(), res.F[:, 1].max()

for level in carbon_levels:
    xx, yy = np.meshgrid(np.linspace(cost_min, cost_max, 100),
                         np.linspace(blow_min, blow_max, 100))
    zz = np.full_like(xx, level)
    ax.plot_surface(xx, yy, zz, alpha=0.2, color='gray')

# 连接点到平面的虚线（与原代码一致）
for cost, blow, carbon in zip(res.F[:, 0], res.F[:, 1], all_carbons):
    ax.plot([cost, cost], [blow, blow], [carbon, 0.021], linestyle='--', color='gray', linewidth=0.5)
    ax.plot([cost, cost], [blow, blow], [carbon, 0.1], linestyle='--', color='gray', linewidth=0.5)

# 标签和标题
ax.set_xlabel('obj1', fontsize=15)
ax.set_ylabel('obj2', fontsize=15)
ax.set_zlabel('Constraint conditions', fontsize=15)
ax.zaxis.labelpad = 20
ax.yaxis.labelpad = 20
ax.xaxis.labelpad = 20
ax.set_title('Optimization result', fontsize=15)

ax.legend(loc='upper right')
ax.grid(False)   # 去掉网格线

# 保存高分辨率图片
plt.tight_layout()
plt.savefig('pareto_front_3d.png', dpi=600)
plt.show()

# ------------------------------
# 6. 评估当前炉次（优化前）
# ------------------------------
current_x = np.array([11693, 4181, 480, 9884, 1721, 207, 0, 1347, 163, 42.85])
current_x_round = np.round(current_x).reshape(1, -1)
current_full = np.concatenate((xd, current_x_round), axis=1)
current_blow = model1.predict(current_full)[0]
current_cost = (current_x_round[0,0]*477/840 + current_x_round[0,1]*554/800 +
                current_x_round[0,2]*750/560 + current_x_round[0,3]*600/1000 +
                current_x_round[0,4]*180/1000 + current_x_round[0,5]*6300/1000 +
                current_x_round[0,6]*6200/1000 + current_x_round[0,7]*19000/1000 +
                current_x_round[0,8]*42400/1000 + current_x_round[0,9]*2200)
current_carbon = model2.predict(current_full)[0]

print("\n=== 当前生产批次（优化前）===")
print(f"目标函数1: {current_cost:.2f} ")
print(f"目标函数2: {current_blow:.2f} ")
print(f"约束变量: {current_carbon:.4f} ")
print("=== 优化后 Pareto 前沿边界 ===")
print(f"最低目标函数1: 目标函数1={res.F[:,0].min():.2f}, 目标函数2={res.F[np.argmin(res.F[:,0]),1]:.2f}")
print(f"最低目标函数2: 目标函数2={res.F[:,1].min():.2f}, 目标函数1={res.F[np.argmin(res.F[:,1]),0]:.2f}")