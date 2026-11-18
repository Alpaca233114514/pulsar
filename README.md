# Pulsar

Pulsar 是一个用手机三轴加速度研究计步的 Python 工具。它关注拄拐、膝关节离断和髋关节离断三类步态，提供手持与大腿佩戴两种位置的合成信号，让你对照足部触地标签，观察算法在何时多算或漏算。你也可以导入自己的手机或 IMU 记录，使用同一套规则做离线分析。

目前的验证限于合成数据。步态和佩戴位置由使用者指定，真实受试者的计步准确性与能耗估算仍待验证。

## 快速开始

需要 Python 3.11 或更高版本，运行时只用标准库。在仓库根目录打开 PowerShell，运行以下命令。每次生成数据都需要一个新目录，示例用时间戳保留每次的结果。

```powershell
$demo = "artifacts/demo-$(Get-Date -Format 'yyyyMMdd-HHmmss')"
python -m pulsar demo --seed 42 --out $demo
Start-Process "$demo/index.html"
```

报告包含 9 个场景。先选「膝关节离断 · 大腿」，对照绿色圆点（模拟器标注的运动周期）和橙色竖线（检测到的事件），再切换到「摆手干扰」，观察有规律的手部运动如何触发计步。完整结果保存在 `report.json`，各场景目录包含传感器数据、标签和分析结果。

下表节选自上述命令（`--seed 42`）。每个场景的参考步数都来自模拟器生成的足部触地标签。

| 合成场景 | 参考步数 | 估算步数 | 如何理解 |
| --- | ---: | ---: | --- |
| 静止 | 0 | 0 | 此段静止噪声未触发计步 |
| 原地摆手 | 0 | 39 | 即使没有迈步，周期性摆手也会被计入 |
| 膝关节离断 · 大腿 | 33 | 34 | 检出 17 个同侧运动周期，按每周期两步换算 |
| 髋关节离断 · 手持 · 长丢包 | 27 | 24 | 数据缺口两侧分开检测，缺口内的触地仍计入参考步数 |

## 分析一段记录

直接复用刚生成的场景即可跑通导入流程。`analyze` 读取传感器文件，将结果以 JSON 打印到终端。

```powershell
python -m pulsar analyze "$demo/knee_disarticulation_thigh/sensors.csv" --source synthetic --gait knee_disarticulation --placement thigh
```

换成手机采集的文件时，按实际情况填写参数。例如，一段手持手机记录的拄拐步行，时间戳单位为毫秒、加速度单位为 g，可这样分析：

```powershell
python -m pulsar analyze phone.csv --gait crutch_assisted --placement hand --timestamp-unit ms --acceleration-unit g
```

| 输入 | 要求 |
| --- | --- |
| 文件格式 | CSV 首行使用列名，JSON 可用记录数组或 `{"samples": [...]}`。自定义列名可用 `--columns` 映射。 |
| 时间 `t` | 数值严格递增。默认秒，也接受 `ms`、`us`、`ns`。采样率建议 50–100 Hz，最低 20 Hz。 |
| 加速度 `ax, ay, az` | 使用**包含重力**的原始三轴加速度。默认 m/s²，单位为 g 时传 `--acceleration-unit g`。 |
| 陀螺仪 `gx, gy, gz` | 可选。如提供，整段记录需包含时间对齐的三轴数据。默认 rad/s，可选 `deg/s`。当前检测器只使用加速度。 |
| 步态与位置 | `--gait` 接受 `normal`、`crutch_assisted`、`knee_disarticulation`、`hip_disarticulation`。`--placement` 接受手持 `hand`、大腿 `thigh`、腰部 `waist`、躯干 `trunk`。 |

### 从信号到步数

检测器先对三轴加速度取模长，分段重采样，抑制尖峰并移除缓慢变化的基线，再按局部幅度、峰突出度和事件间隔筛选峰值。至少三个连续且符合节律条件的事件才构成活动片段。滤波使用居中窗口，需要读取后续样本，适合整段记录的离线分析。

一次足或假肢足触地计一步，拐杖触地单独标注。手机测到的是佩戴位置的运动，因此 `event_times_s` 中的事件要按位置换算成 `step_count`：

- **手持、腰部、躯干**默认每个事件计一步。摆手和拐杖冲击都可能造成误报。
- **大腿**默认每个同侧运动周期计两步，前提是双侧交替迈步。单腿跳行或摆过式拄拐需要另行标定。可用 `--event-mode step` 或 `--event-mode stride` 显式指定换算方式。

`cadence_spm` 是活动片段内的估算步频（步/分钟），`active_duration_s` 是各检测片段的时长之和。评估时，步数误差与所有足部触地比较，事件 F1 则按 ±0.2 秒容差将检测事件与标注的运动周期一对一匹配。

## 生成合成数据

```powershell
$run = "artifacts/experiment-$(Get-Date -Format 'yyyyMMdd-HHmmss')"
python -m pulsar simulate --gait hip_disarticulation --placement thigh --duration 60 --seed 42 --out "$run/single"
python -m pulsar dataset --subjects 15 --sessions 2 --out "$run/dataset"
```

单段记录分为 `sensors.csv`（传感器读数）、`labels.json`（足部触地、拐杖触地和运动周期）和 `metadata.json`（生成参数与种子）。模拟器会改变步间隔、左右侧幅度和噪声，三类步态的参数均为实验假设。

批量示例生成 45 段记录，每位虚拟受试者有 2 段步行和 1 段负样本。同一人的记录归入同一个分区，本例训练、验证、测试集分别有 27、9、9 段。`manifest.jsonl` 列出每段记录及其分区，`windows.csv` 提供 5 秒非重叠窗口。训练时使用 `summary.json` 中 `training_features` 列出的四项信号特征，以 `target_steps` 为步数标签。窗口特征使用整段离线滤波后的信号。

## 估算能耗

```powershell
python -m pulsar calories --weight-kg 70 --duration 1800 --met 3 --met-range 2 5
```

`--duration` 以秒为单位。以 70 kg、30 分钟、3 MET 为例，按 `MET × 体重（kg）× 时长（小时）` 估算，总能耗为 **105 kcal**，扣除 1 MET 的静息基线后，活动能耗为 **70 kcal**。`--met-range 2 5` 给出不同强度假设下的估算范围。MET 由使用者提供，尚未按步态或个体校准。

若希望基于传感器记录估算能耗，可给 `analyze` 同时传入 `--weight-kg` 和 `--met`。此时计算使用检测到的活动时长，摆手误报和丢包漏检都会影响结果。

## 开发

运行测试或查看各命令的参数：

```powershell
python -m unittest discover -s tests -v
python -m pulsar --help
python -m pulsar analyze --help
```

Python 调用入口为 `pulsar.io.load_recording`（文件）或 `pulsar.io.from_rows`（采集端记录），再将返回的 `Recording` 交给 `StepCounter.analyze`。检测参数见 [algorithm.py](pulsar/algorithm.py)，合成标签与指标的匹配规则见 [evaluation.py](pulsar/evaluation.py)。

项目采用 [Apache License 2.0](LICENSE)。
