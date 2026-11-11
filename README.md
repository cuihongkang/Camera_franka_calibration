# Panda 关节形变与扭曲标定（eye-in-hand 棋盘格）

机械臂用久了，关节附近的连接件会出现扭转、偏移，加载后关节还会弹性变形。结果是用标称 URDF 做 FK/IK 时，实际到达位置和预设位置对不上。本项目只用腕部相机拍桌面上固定的棋盘格，标定出这些误差，并导出补偿后的模型：

- **几何误差**：每个关节原点后接一个 6 自由度刚体误差 `E_i`（3 平移 + 3 旋转，含零位偏置），法兰处同样有一个。
- **关节柔性**：`q_link = q_encoder − c_i · τ_i`。`τ` 是测得的关节力矩（Franka `tau_J`），`c_i = 1/K_i` 是关节柔度。
- 同时估计手眼外参、棋盘格在基座下的位姿和相机内参（含畸变）。

残差是棋盘格角点的重投影误差（像素）。冗余参数由可辨识性分析自动固定，剩下的正好是完备、最小的 4R+6 = 34 个几何参数：基座和法兰误差被棋盘位姿、手眼外参吸收；每个关节沿/绕自身轴的误差会转移到下一个关节。无法激励的柔度（关节 1 轴竖直、关节 7 几乎无负载）也会被固定，并在结果里列出。

输出两样东西：

1. `panda_calibrated.urdf`：修正了各关节 origin，并加了 `camera_optical_frame`，可以直接换给现有的 IK / MoveIt。
2. `calibration.json`：几何误差、柔度、手眼、内参，以及每个参数的标准差。柔性补偿不在 URDF 里，指令关节角要过一次 `panda_calib.ik.compensate_compliance()`。

## 环境

```bash
conda create -y -p "$PWD/.envs/calib" python=3.11
.envs/calib/bin/pip install -r environment/calib-pip.txt   # numpy, scipy, opencv-python-headless<5
python scripts/fetch_franka_assets.py                     # 仅仿真需要：Isaac 5.1 Franka 资产，放到 assets/isaac_5_1
```

下文 `PY` 指 `PYTHONPATH=$PWD .envs/calib/bin/python`。Isaac Sim 本机入口是 `/home/cuihongkang/isaacsim-5.1.0/python235.sh`（glibc 兼容启动器），其他机器换成自身的 `python.sh`。

## 真机流程

1. **棋盘格**：每边格子数一奇一偶（默认 12×9 格、11×8 内角点、30 mm）。贴平、刚性固定，相机采集期间不能动。角点编号由"(0,0) 外侧格子为黑色"加手性检查唯一确定，不会出现 180° 翻转。
2. **规划位姿**：只用标称模型和粗略的棋盘位置、相机安装位置，筛选出满足三个条件的位姿：整块棋盘在视野内（带边距）、关节不越限、连杆高于棋盘。

   ```bash
   $PY scripts/plan_poses.py --output configs/poses_real.json --count 150 \
       --board-xyz 0.42 -0.15 0.0 --mount-xyz 0.065 0 0.05 --fx 615 --image-size 640 480
   ```

   规划**不检查自碰撞，也不检查和桌面其他物体的碰撞**。执行前请在 MoveIt / 示教器里确认路径安全。
3. **采集**：每个位姿到位后**静止约 1 s**，记录编码器角 `q`（libfranka `q`）、测得力矩 `tau_J` 和一张图像，存成：

   ```
   data/<name>/samples.npz      q (N,7) rad, tau (N,7) N·m, image_size (w,h)
   data/<name>/images/000000.png …
   data/<name>/metadata.json    {"board": {"cols": 11, "rows": 8, "square": 0.03}}
   ```

   **相机要关掉自动对焦**，不同距离的焦距会变。曝光要保证角点清晰、无运动模糊。
4. **标定**：

   ```bash
   $PY scripts/calibrate.py data/<name>          # 默认每 4 个视角留出 1 个做测试
   ```

   默认同时跑三种模型，用来对比：
   - `handeye`：标称运动学，只标手眼 / 内参
   - `geometry`：加上几何误差
   - `full`：几何误差加柔性

   每种模型的结果写到 `results/<name>/<mode>/`。留出视角的重投影误差应当接近角点检测噪声。如果 `full` 明显好于 `geometry`，说明柔性确实存在。
5. **使用**：

   ```python
   from panda_calib.export import load_result
   from panda_calib.ik import solve_ik, compensate_compliance
   from panda_calib.kinematics import PandaModel
   cal = load_result('results/<name>/full/calibration.json')
   model = PandaModel.from_urdf(extra_hand_mass=0.3, extra_hand_com=[.065, 0, .03])  # 末端实际负载
   q_link, ok, _ = solve_ik(model, T_base_hand_target, q_ref, cal['geo'])
   q_cmd = compensate_compliance(model, q_link, cal['compliance'], cal['geo'])        # 发给机器人的指令
   ```

   相机检测到的目标点要用标定后的模型转到基座系：`model.fk(q_link, cal['geo']) @ cal['T_hand_cam']`，其中 `q_link = link_side_angles(q, tau_J, cal['compliance'])`。`compensate_compliance` 默认用 URDF 质量模型估计目标位姿的重力矩，也可以传入实测 `tau` 迭代一次。

## 仿真验证（Isaac Sim 5.1）

仿真里的 Franka 就是"真机"：关节坐标系里写入已知的随机误差，关节驱动当作带刚度的柔性关节，在重力下自然变形。腕部 RGB 相机按 RTX 渲染出棋盘格。标定只读 `samples.npz` 和图像，真值单独存在 `truth.json` / `truth.npz`。

```bash
$PY scripts/plan_poses.py --output configs/poses_sim.json --count 300 --seed 0
/home/cuihongkang/isaacsim-5.1.0/python235.sh "$PWD/scripts/sim_collect.py" --output "$PWD/data/sim_seed0" --gpu 1
$PY scripts/calibrate.py data/sim_seed0
$PY scripts/evaluate_sim.py data/sim_seed0
/home/cuihongkang/isaacsim-5.1.0/python235.sh "$PWD/scripts/sim_ik_demo.py" "$PWD/data/sim_seed0" --gpu 1
```

- `evaluate_sim.py`：在留出视角上，把各模型预测的相机位姿、法兰（TCP）位姿和渲染真值对比，并对比柔度和注入值。
- `sim_ik_demo.py`：完整复现"看到目标 → IK → 到达"。用留出视角的检测结果估计棋盘位置，把 TCP 送到某个角点上方 2 cm，再从物理引擎读 TCP 实际位置，统计误差。

结果见 [实验报告](reports/sim_report.md)。

## 验证

```bash
$PY -m unittest discover -s tests -v
```

测试覆盖以下内容：
- FK 与 Franka 官方 MDH 参数表一致
- 重力力矩等于势能梯度
- 雅可比矩阵与有限差分一致
- IK 收敛
- 柔性补偿与静平衡互逆
- 导出 URDF 的 FK 与标定模型一致
- 投影与 `cv2.projectPoints` 一致
- 任意面内旋转下角点编号唯一
- 合成数据上参数恢复，冗余参数个数为 4R+6

## 局限

- **TCP 和相机的相对位置必须另外已知。** eye-in-hand 只能观测相机。关节 7 沿轴的平移误差、法兰误差会被手眼外参吸收，相机预测仍然准确，但 TCP 会带着这部分误差。
- **柔性模型只含绕关节轴的扭转。** 连杆弯曲、减速器背隙、和转向相关的误差都没有建模。磨损严重时，背隙可能是主要误差来源，此时 `full` 模式的留出残差会明显高于检测噪声。
- **仿真只是一次已知误差的自洽验证。** 仿真图像无运动模糊、无镜头畸变，内参恒定，验证的是方法和代码，不代表真机精度。真机上要特别注意对焦、曝光和棋盘平整度。
- **重力补偿依赖 URDF 的质量参数。** URDF 质心都在连杆原点，比较粗糙。真机上建议用实测 `tau_J` 迭代补偿，或者换成 Franka 辨识出的动力学参数。
