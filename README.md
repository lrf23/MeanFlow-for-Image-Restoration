﻿# MeanFlow-for-Image-Restoration

## 介绍


尝试通过修改流速度表达式，将MeanFlow模型应用在通用图像修复领域。

首先MeanFlow在图像生成上的流速度表达式如下（ [Mean Flows for One-step Generative Modeling](https://arxiv.org/abs/2505.13447)）：

```math
v_t=\epsilon-x \quad (1)
```

为尝试将其用于通用图像修复，尝试了**两种流速度表达式**。

第一种表示如下：

```math
v_t=x_{in}-x \quad (2)
```

其中$`x_{in}`$为退化图像（即待修复图像），$`x`$为真实图像，个人理解的就是，Flow Matching本身是真实分布和目标分布两者之间的转换，前向过程是从真实分布（比如猫狗图像）流到目标分布（一般为高斯噪声），而后向过程是从目标分布流到真实分布。那么对于图像修复而言，真实分布应该是高质量图像，目标分布应该是退化图像，前向后向过程同理，因此能够得到公式（2）。

第二种表示如下：

```math
\begin{equation}
v_t=(1-\delta)x_{in}-x+b\epsilon  \quad (3)
\end{equation}
```

其中$`\delta，b`$为常数项，控制退化分布和噪声分布的比例。$`\epsilon`$为高斯噪声。其中公式（2）是公式（3）的$`\delta=0,b=0`$时的特殊情况。这两公式表达的含义相同，都是建模退化分布和真实分布之间的差值。公式（3）的$`\delta`$项是为了让融合多种任务的退化分布，b项是为了加个噪声，可能可以提升图像修复的多样性？该公式是参考了 [DiffUIR](https://arxiv.org/abs/2403.11157) 的共享分布项原理得到的，**其正确性有待考究**。

`src/model_meanflow_v1`里用的是公式（2），`src/model_meanflow_v2`里用的是公式（3）。`src/UnetRes_Meanflow`为预测平均流速度的网络。用公式（2）训练的效果还挺好。

## 数据集

关于数据集路径问题请查阅[universal_dataset.py](https://github.com/lrf23/MeanFlow-for-Image-Restoration/blob/main/data/universal_dataset.py)文件

关于数据集下载请参考[iSEE-Laboratory/DiffUIR: (CVPR2024) Official implementation of paper: "Selective Hourglass Mapping for Universal Image Restoration Based on Diffusion Model"](https://github.com/iSEE-Laboratory/DiffUIR?tab=readme-ov-file) 

本人仅在去模糊、去雨、去雪、低光修复四个任务上进行了训练和测试。

## 结果展示

对模型的详细说明和结果展示如下图所示：

![poster](https://github.com/lrf23/MeanFlow-for-Image-Restoration/blob/main/Image/poster.png)

## 其他

有任务问题欢迎在issue处交流指正😊😊😊
