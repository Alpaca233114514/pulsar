# Pulsar

面向更多人的计步算法（研究中）。Python demo 支持**拄拐、膝关节离断、髋关节离断**三类步态配置，优先处理手机**手持、大腿部**数据，预留腰部与躯干 IMU 接口，并提供降噪、合成数据生成、评估和卡路里估算。

当前是**离线、规则驱动的实验基线**。三类步态参数是可调整的工程假设，不代表已对真实患者验证；模拟器是现象级信号生成器，不是肌骨动力学仿真。尚未训练模型，也不自动判断截肢类型。

## 直接运行

需要 Python 3.11+，运行时仅使用标准库，无需安装依赖。以下命令在仓库根目录执行：

```powershell
# 生成九个场景、降噪波形、真实标签、算法结果和交互式能耗 demo
python -m pulsar demo --out artifacts/demo
Start-Process artifacts/demo/index.html

# 生成一段手机大腿部信号
python -m pulsar simulate --gait hip_disarticulation --placement thigh --duration 60 --out artifacts/hip-thigh

# 批量生成训练样本：默认 15 个虚拟受试者，每人 2 段步行 + 1 段负样本
python -m pulsar dataset --out artifacts/training --subjects 15 --sessions 2

# 对生成数据走同一传感器导入接口
python -m pulsar analyze artifacts/hip-thigh/sensors.csv --source synthetic --gait hip_disarticulation --placement thigh --weight-kg 70 --met 3

# 独立能耗演示：70 kg、30 分钟、3 MET；2–5 MET 是场景范围
python -m pulsar calories --weight-kg 70 --duration 1800 --met 3 --met-range 2 5

python -m unittest discover -s tests -v
```

输出目录必须是新目录；重复运行请换一个 `--out`，避免覆盖已有数据。生成文件位于被 Git 忽略的 `artifacts/`，真实传感器数据也建议保存在其中。

## 算法与降噪

1. 校验时间戳、单位和传感器位置；对短丢包和时间抖动重采样。超过 `max(0.15 秒, 3.5 × 中位采样间隔)` 的长缺口断开处理。
2. 对加速度取模长，减少坐标朝向影响；Hampel 滤波抑制孤立尖峰。
3. 约 80 ms 居中滑动平均平滑高频噪声，减去约 2.5 s 的局部基线，去除重力常量和慢漂移。
4. 使用局部自适应幅度阈值、峰突出度、不应期、最小连续事件数和宽松的节律约束检测运动事件。三类步态允许更大的间隔不对称性；规则参数可通过 `DetectorConfig` 调整。
5. 输出事件时间、估算步数、步频、间隔变异系数、活动时间、丢包区间和降噪诊断。短于 3 个连续运动事件的片段不计数。

**计数口径**：脚或假肢足的一次触地算一步，拐杖触地不算脚步。当前只能从手机信号推断运动代理事件，无法保证每个峰就是脚触地。手持、腰部、躯干默认按一事件一步；大腿默认按一运动周期两步，结果明确记录 `stride_proxy_x2`。可用 `--event-mode step/stride` 覆盖。大腿模式仅适用于交替步态，单腿跳行、摆过式拄拐、无假肢足触地、双侧截肢等需要另外定义计数口径并重新标定，当前不宣称支持。

降噪采用离线居中窗口，会使用后续样本，**不是实时流式算法**。静态朝向旋转不改变模长，但动态拿放手机、摆手、周期性颠簸与拐杖冲击仍可能误计数。demo 保留零脚步的摆手负样本以显示这一限制。陀螺仪数据已可导入和生成，本版检测器尚未融合陀螺仪。

## 真实手机 / IMU 数据接口

传入**包含重力的原始三轴加速度**，默认 m/s²。不要直接把 Android/iOS 的 linear/user acceleration（去重力数据）当作 raw accelerometer。建议 50–100 Hz，当前最低支持 20 Hz。数据必须来自同一个采样时钟，时间戳严格递增；不同传感器的异步采样需要采集端先同步。

CSV 必需列为 `t,ax,ay,az`，可选列为 `gx,gy,gz`（三轴必须同时提供，默认 rad/s）。JSON 支持记录数组或 `{"samples": [...]}`。时间戳是数值型 s/ms/us/ns，导入时归一到起始时刻。来源默认标为 `real`，合成输入请显式传 `--source synthetic`。

```powershell
python -m pulsar analyze artifacts/phone.csv --gait knee_disarticulation --placement hand --timestamp-unit ms --acceleration-unit g --gyro-unit deg/s --columns '{"t":"time","ax":"acc_x","ay":"acc_y","az":"acc_z"}' --weight-kg 70 --met 3
```

Python 接口可直接接采集端、文件或其他设备适配器：

```python
from pulsar import DetectorConfig, Gait, Placement, StepCounter, estimate_calories
from pulsar.io import from_rows

# rows 为设备适配器输出的字典迭代器；当前算法整段离线运行。
recording = from_rows(rows, placement="hand", timestamp_unit="ms")
result = StepCounter(DetectorConfig(gait=Gait.CRUTCH, placement=Placement.HAND)).analyze(recording)
calories = estimate_calories(
    weight_kg=70, duration_s=result.active_duration_s, met=3,
    duration_source="detected_bouts",
)
```

结果中的 `cadence_spm` 按事件簇内部的平均间隔换算，避免左右不对称时仅取中位数造成偏差；`interval_cv` 包含步态本身的不对称性，不能解释为病理评分。`active_duration_s` 按事件簇边界估算，排除长丢包，不能当作真实活动时长标注。

## 合成数据与训练接口

三类配置使用不同的名义步频、不对称性和躯干侧向变化。拄拐额外生成独立拐杖触地冲击；大腿模式主要在同侧脚接触时生成运动周期。模拟默认交替足/假肢足触地，左右标记仅为虚拟侧别，未建模实际患侧。

随机化包含步频、步间抖动、左右幅度差、传感器固定旋转、偏置、加性噪声、尖峰、随机丢包和采样抖动；Python `SimulationConfig` 还支持停顿、长丢包、指定步频与采样率。seed 可复现。随机参数并不保证生物力学真实性，也没有建立长期受试者身份模型。

每段输出：

- `sensors.csv`：纯传感器输入，不包含标签。
- `labels.json`：`foot_contact`、`crutch_contact` 和 `motion_cycle` 的独立时间标签。
- `metadata.json`：随机种子、生成参数、方向和计数假设。

批量模式另有 `manifest.jsonl`、`windows.csv`、`summary.json`。默认按虚拟受试者分为 train/validation/test = 60/20/20，每个受试者的所有片段属于同一分区；15 人的完整分块保证每个分区包含三类步态，人数更少时可能缺分区。5 秒非重叠窗口提供绝对均值、标准差、RMS、峰峰值，以及真实脚步数标签 `target_steps`。仅将 `summary.json` 的 `training_features` 用作数值特征，避免把目标或标签字段误作输入。特征基于整段离线滤波，不适合直接用于实时性能评估。

这一步交付的是**生成器、可训练数据接口和规则基线**；没有把同一模拟器上的指标当作模型泛化证据。评估同时给出步数误差和 ±0.2 秒容差的一对一事件 precision/recall/F1；大腿匹配真实运动周期，步数仍和所有足触地比较。丢失区间内的真值保留，因此漏计会计入误差。无真实脚步的负样本单独报告误计步数。

## 卡路里模型

总能耗（kcal）≈ `MET × 体重 kg × 活动小时`；高于静息的活动能耗 ≈ `max(MET − 1, 0) × 体重 kg × 活动小时`。例如 70 kg、30 分钟、3 MET：总能耗约 **105 kcal**、活动能耗约 **70 kcal**。

1 MET 约等于 1 kcal/kg/hour，定义来自 [Compendium of Physical Activities](https://pacompendium.com/)。MET 并非个体精确能耗，参见其[方法说明](https://pacompendium.com/corrected-mets/)。本版不根据截肢类型或步数直接套用额外倍率；`--met` 由调用者提供，demo 中的 3 MET 只是可修改的假设。`met_range` 产生的是场景范围，**不是统计置信区间**。体重口径、假肢质量、真实活动强度等需要真实测量与能耗实验校准。

## 开源方案检索与后续验证

实现前已检索 [gaitmap](https://github.com/mad-lab-fau/gaitmap)、[OpenSim](https://github.com/opensim-org/opensim-core) 和手机计步项目 [the-pedometer-algorithm](https://github.com/rcbao/the-pedometer-algorithm)。前两者分别侧重足部 IMU 分析和肌骨仿真；本版没有复制这些项目代码，也没有新增第三方运行依赖。

接下来需要采集目标人群在手持、大腿、腰部/躯干位置的同步信号，以视频或足底接触标注校准事件口径，补充转身、扶拐、拿放手机、乘车、摆手等负样本；按真实受试者分组评估，才能判断计步适用范围。能耗模型需要独立的参考测量，计步测试通过不能验证卡路里准确性。
