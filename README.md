# 交接文档

## 一、交接内容

- 实验内容
  - 实验代码
  - 实验结果
    - 《重车实验汇总》：所有有用结果的记录，大量无用实验结果已删除
- 毕业论文
  - 毕业论文
  - 答辩 PPT
- 交接文档
  - README.md



## 二、实验环境（推荐使用 conda 来搭建环境）

重要包如下表所示。

如有遗漏，按报错提示安装所需的包即可。

|    包/库/框架    |     版本      |
|:------------:|:-----------:|
|    python    |   3.10.12   |
|    numpy     |   1.26.3    |
|    pandas    |    2.1.4    |
| scikit-learn | 1.4.1.post1 |
|    torch     |    2.0.1    |
| tensorboard  |   2.16.2    |
|    einops    |    0.7.0    |
|  matplotlib  |    3.8.3    |
|     timm     |   0.9.16    |



## 三、复现实验

### 1、代码结构

代码结构如下。

- src
  - data
    - PlanarBeam
    - ...
  - datasets
    - data.py
    - dataset.py
    - ...
  - metrics
    - damage_mae.py
    - ...
  - models
    - Modules
      - GRL.py
      - ...
    - PreBDI.py
    - ...
  - utils
    - analysis.py
    - ...
  - main.py
  - options.py
  - running.py
  - example.sh
  - ...

### 2、工程说明

`src`下的`data`文件夹包含处理后的数据，可以直接用于实验，`datasets`包含数据处理的代码，`metrics`包含设计的损伤实验指标，`models`包含实验模型代码，`utils`中包含可视化分析代码。

`src`下的`main.py`是实验入口，可通过编写shell文件运行`main.py`,实验所需的各项参数在`options.py`文件中说明，包括模型的数据集设置，参数设置，优化器设置，损失函数设置等，各项参数均能按照数据集自行设置。

|                      文件                      |     内容     |
|:--------------------------------------------:|:----------:|
|  `bridge_damage_detection / src / main.py`   |  模型相关实验入口  |
| `bridge_damage_detection / src / options.py` | 模型相关实验参数配置 |
| `bridge_damage_detection / src / running.py` |   模型运行文件   |
| `bridge_damage_detection / src / example.sh` |  模型实验示例文件  |

