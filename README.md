# 某临海城市实时指控

这个目录可以单独拿走。对方用自己的 AFSIM 2.9 和自己的 Python，选红方阵地和一套蓝方航线，一条命令启动。Warlock 窗口就是这场仿真。周视雷达出现航迹之后，神经网络开始点名。

## 里面有什么

- `network/`：十二平台分配网络，方案 `l1.coastal-allocate/1.2`。上场的权重是种子 3，在 `network/weights/seed_3.pt`。
- `gateway/`：实时接口网关。仿真把平台状态和雷达航迹写成表，网关读表、调用网络，再把点名写成 `net_1.txt`、`net_2.txt`。场景里原来的距离门会被换成读这些文件的脚本。
- `scenario/`：这场能在 AFSIM 里打开所必需的场景和平台类型。类型文件里保留了仿真要读的距离和效果，没有另外写一份交战说明。
- `plans/fixed.json`：红方默认方案，某临海城市固定阵地。只有驻留，不画航线。
- `plans/` 里另外四份是蓝方，都在末端撞击：
  - `leap.json` 跃起俯冲撞击（默认）
  - `wave.json` 波浪下降撞击
  - `skim.json` 贴地拉起再砸
  - `stairs.json` 错层阶梯撞击

平台名字不要改。权重按这些名字对齐：`laser_1`、`laser_2`、`net_1`、`net_2`、`hunter_1`、`hunter_2`、`jam_1`、`jam_2`、`eo_1`、`eo_2`、`radar_1`、`radar_2`。两部雷达不分配任务。

## 部署

需要 Python 3.11 和 PyTorch。推理用 CPU 就够。AFSIM 用对方自己的 2.9，目录里要有 `bin/warlock.exe`。这个包不带 AFSIM。

```bash
pip install -r requirements.txt
python run_live.py --afsim "D:\路径\afsim-2.9.0-win64" --blue plans/leap.json
```

也可以先设环境变量 `AFSIM_ROOT`，再省略 `--afsim`。红方不写时用 `plans/fixed.json`。换蓝方只改 `--blue`，例如 `plans/wave.json`。

启动后大约 8 秒，十二个平台都有状态，并且周视雷达 `radar_2` 在最近 3 秒里有航迹，才会打出第一拍。之后每拍至少隔 2 秒。点名记录写在这场目录的 `ticks.jsonl`。关掉 Warlock，或在终端按 Ctrl+C，都会停。

## 这套网络学了什么

编码器不改。每个平台在航迹上做一个指针。威胁几何是相对东、相对北，以及这两个数的平方，单位是公里，再加 0.1 倍编码器输出。

规则老师的标签是：离保护点更近的航迹排在前面；每个武器只在自己的作用距离里选；激光已经认领的航迹，干扰让开。这不是“谁离武器近就打谁”。

种子 3 过了考核桌：指针 0.8177，排序对 0.8519，对调场景里 12 个平台都跟对。考核线是指针不低于 0.75，排序对不低于 0.75，对调后至少 9 个平台跟对。

任务种类由平台决定，不由网络另选：激光、网、撞击是 hard，干扰是 soft，光电是 observe。一条命令的样子是 `laser_1 HARD 航迹名 编号`。

## 换成自己的模型

只换分配函数。输入仍是这一拍的十二个平台状态，加上最近的周视雷达航迹，坐标是相对东、北。输出仍是每个平台一个航迹编号，或者空着。

看 `examples/your_model.py` 里的 `assign`。接上它：

```bash
python run_live.py --afsim "D:\路径\afsim-2.9.0-win64" --policy examples.your_model:assign
```

网关继续负责写命令文件。自己的函数返回：

```text
{"assignments": [{"platform": "laser_1", "token": "radar_2:名字.编号", "task": "hard"}, ...]}
```

空的 `token` 表示这一拍这个平台不动。

## 行为克隆和泛化

每一拍的输入和分配都在 `ticks.jsonl`。想用规则老师重新学一遍：

```bash
python train/clone.py
```

新权重写到 `train/out`，不会替换 `network/weights/seed_3.pt`。打印出的指针、排序对、对调跟对，三项都过线，才可以自己换上。

## 强化学习

整场分数是：蓝方被去掉的数量，减去保护点被撞击次数的 10 倍。见 `train/score.py`。

现在这套强化学习只在一份拷贝上做一步 Adam，学习率 `1e-5`。过不了上面那张考核桌就丢掉。它不是上场的策略。上场的仍然是种子 3。
