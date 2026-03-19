# 第四章 基于执行引导的偏好优化方法

## 4.1 问题定义与方法概述

在结构化幻灯片数据到HTML代码的自动生成任务中，模型需要将包含元素坐标、文本内容及图片路径的JSON格式输入转换为可直接在浏览器中渲染的完整HTML文档。该任务的核心挑战在于：生成的HTML不仅需要在语法层面正确可执行，还需要在布局层面忠实还原输入数据所描述的空间关系，同时在视觉层面具备丰富的样式表现力。

传统的监督微调（Supervised Fine-Tuning, SFT）方法依赖于人工标注的参考HTML作为训练目标，但HTML代码的表达空间极为庞大——同一视觉效果可由截然不同的代码实现——这使得逐token的交叉熵损失难以有效捕捉生成质量的本质差异。此外，人工标注高质量HTML-幻灯片对齐数据的成本极高，严重制约了训练数据的规模化获取。

针对上述问题，本章提出基于执行引导的偏好优化方法（Execution-Guided Preference Optimization, EGPO）。该方法的核心思想是：利用浏览器渲染引擎作为执行环境，对模型生成的候选HTML进行实际渲染，并从渲染结果中自动提取多维度的质量评估指标，进而构建偏好训练数据，通过直接偏好优化（Direct Preference Optimization, DPO）对模型进行对齐训练。

EGPO方法的整体流程包含以下四个阶段：

（1）候选生成阶段：对每个输入幻灯片，通过多次随机采样从基座模型生成N个候选HTML文档；

（2）执行评估阶段：利用Playwright无头浏览器渲染每个候选HTML，通过JavaScript注入提取DOM元素的实际渲染坐标、文本内容及视觉属性，计算多维度评估指标；

（3）偏好对构建阶段：基于双轨评分体系的结果，从候选集中选取得分最高者作为chosen样本、得分最低者作为rejected样本，构成偏好训练对；

（4）DPO微调阶段：利用构建的偏好数据集，采用LoRA高效微调策略对基座模型进行DPO训练，其中训练信号由独立设计的reward函数提供。

与现有方法相比，EGPO具有以下优势：第一，评估过程完全自动化，无需人工标注；第二，评估基于实际渲染结果而非静态代码分析，能够捕捉代码执行后的真实视觉效果；第三，通过双轨评分体系的设计，实现了偏好对选择（score）与训练信号引导（reward）的解耦，使模型在保证正确性的同时向更丰富的视觉表现力演进。

## 4.2 候选HTML生成策略

候选生成是EGPO流程的起点。对于每个输入幻灯片 $s_i$，本方法从基座语言模型 $\pi_\theta$ 中通过多次独立采样生成 $N$ 个候选HTML文档：

$$\{h_i^{(1)}, h_i^{(2)}, \ldots, h_i^{(N)}\} \sim \pi_\theta(\cdot \mid p(s_i))$$

其中 $p(s_i)$ 为根据幻灯片 $s_i$ 的blocks数据构造的提示文本（prompt）。

提示文本的设计遵循四步流程范式，引导模型依次完成布局结构感知、现代CSS布局实现、配色方案设计和装饰元素添加。提示中包含以下硬性约束：

（1）画布尺寸约束：幻灯片画布固定为1920px × 1080px，要求在HTML/CSS中显式体现；

（2）布局方式约束：禁止使用 `position:absolute` 或 `position:fixed` 进行内容元素的布局定位，必须优先使用Flexbox与CSS Grid实现布局；

（3）元素映射约束：要求为每个输入元素生成对应的DOM元素，并通过 `data-json-id` 属性建立从DOM元素到输入数据的精确映射关系。

在采样策略上，本方法采用温度采样（temperature sampling）结合top-p截断的方式。默认温度参数 $\tau = 0.7$，top-p参数 $p = 0.9$。较高的温度设置旨在增加候选集的多样性，使得不同候选在布局策略、配色方案和装饰风格上产生差异，从而为后续的偏好对构建提供充分的对比空间。每个候选的最大生成token数设为4096，以确保能够容纳完整的HTML文档。

生成完成后，通过正则表达式从模型输出中提取完整的HTML文档片段。提取逻辑优先匹配 `<!DOCTYPE html>` 声明，其次匹配 `<html` 标签起始位置，最后回退到去除Markdown代码块标记的纯文本。

## 4.3 基于执行反馈的多维评估指标体系

本方法设计了七个互补的评估维度，从不同角度衡量生成HTML的质量。这些指标的计算均依赖于Playwright无头浏览器的实际渲染结果，而非对HTML源代码的静态分析。以下逐一阐述各指标的定义与计算方法。

### 4.3.1 渲染有效性（Render Validity）

渲染有效性是所有后续指标计算的前提条件，作为整个评分体系的硬门槛。本方法通过Playwright将候选HTML加载到1920×1080视口的无头Chromium浏览器中，并检测以下条件：

（1）页面加载过程中是否触发JavaScript运行时错误（通过监听 `pageerror` 事件捕获）；

（2）页面是否存在可见的主容器元素（`.slide` 类或 `body` 元素具有正的宽高）；

（3）页面是否包含任何可见内容（DOM节点、文本或图片）。

当上述任一条件不满足时，该候选HTML被判定为渲染无效（`valid_render = False`），其score直接置为0，reward直接置为-1，不再计算其余指标。这一硬门槛设计确保了进入偏好对的chosen样本必然是可正常执行和渲染的HTML文档，从根本上保证了训练数据的基本质量。

### 4.3.2 布局对齐度（Layout Alignment）

布局对齐度衡量生成HTML中各元素的实际渲染位置与输入数据中指定坐标的吻合程度。本方法采用交并比（Intersection over Union, IoU）作为度量指标。

对于输入数据中的每个目标元素 $e_k$，其参考矩形框为 $R_k^{ref} = (x_k, y_k, w_k, h_k)$，通过Playwright渲染后提取的实际矩形框为 $R_k^{pred}$。两者的IoU定义为：

$$\text{IoU}(R_k^{ref}, R_k^{pred}) = \frac{|R_k^{ref} \cap R_k^{pred}|}{|R_k^{ref} \cup R_k^{pred}|}$$

其中 $|\cdot|$ 表示矩形面积。交集区域的计算通过坐标裁剪实现：

$$x_1^{inter} = \max(x_k^{ref}, x_k^{pred}), \quad y_1^{inter} = \max(y_k^{ref}, y_k^{pred})$$
$$x_2^{inter} = \min(x_k^{ref} + w_k^{ref}, x_k^{pred} + w_k^{pred}), \quad y_2^{inter} = \min(y_k^{ref} + h_k^{ref}, y_k^{pred} + h_k^{pred})$$

当交集区域的宽或高为非正值时，IoU为0。布局对齐度取所有成功匹配元素的IoU均值：

$$\text{LayoutAlign} = \frac{1}{|M|} \sum_{k \in M} \text{IoU}(R_k^{ref}, R_k^{pred})$$

其中 $M$ 为成功匹配的元素集合，即在渲染结果中找到对应 `data-json-id` 的元素。该指标的取值范围为 $[0, 1]$，值越高表示布局还原越精确。

实际渲染坐标的提取通过向页面注入JavaScript代码实现。注入脚本遍历所有带有 `data-json-id` 属性的DOM元素，调用 `getBoundingClientRect()` 获取其相对于幻灯片容器的坐标偏移，并过滤掉不可见元素（`display:none`、`visibility:hidden` 或零尺寸元素）。

### 4.3.3 DOM覆盖率（DOM Coverage）

DOM覆盖率衡量输入数据中的目标元素在生成HTML中被成功映射的比例。其定义为：

$$\text{DOMCoverage} = \frac{|M|}{|T|}$$

其中 $|T|$ 为输入数据中目标元素的总数，$|M|$ 为在渲染结果中成功找到对应DOM元素的数量。目标元素的定义遵循以下规则：若某个block存在children，则以每个child作为目标元素；若block没有children，则以block自身作为目标元素（仅当其包含可见文本或图片时）。

DOM覆盖率在本方法的评分体系中扮演特殊角色——它不作为加法项参与评分，而是作为乘法门控因子（详见4.4节）。这一设计的动机在于：DOM覆盖率是布局对齐度等其他指标的前置条件。如果大量元素缺失，即使已匹配元素的IoU很高，整体生成质量也不应获得高分。将覆盖率作为乘法门控可以有效惩罚元素缺失的情况。

### 4.3.4 文本保留度（Structure Validity）

文本保留度衡量生成HTML中各元素的文本内容与输入数据中原始文本的一致性。本方法采用ROUGE-L F1分数作为度量指标，该指标基于最长公共子序列（Longest Common Subsequence, LCS）计算。

对于每个包含文本的目标元素，设参考文本的token序列为 $\mathbf{r} = (r_1, r_2, \ldots, r_m)$，生成HTML中对应元素的文本token序列为 $\mathbf{c} = (c_1, c_2, \ldots, c_n)$。ROUGE-L F1定义为：

$$R_{lcs} = \frac{\text{LCS}(\mathbf{c}, \mathbf{r})}{|\mathbf{r}|}, \quad P_{lcs} = \frac{\text{LCS}(\mathbf{c}, \mathbf{r})}{|\mathbf{c}|}$$

$$\text{ROUGE-L}_{F1} = \frac{2 \cdot R_{lcs} \cdot P_{lcs}}{R_{lcs} + P_{lcs}}$$

其中 $\text{LCS}(\mathbf{c}, \mathbf{r})$ 为两个序列的最长公共子序列长度。考虑到输入数据中可能包含中文文本，本方法采用jieba分词工具进行中文分词，将文本切分为词级token序列后再计算ROUGE-L。当jieba不可用时，回退到字符级切分。

文本保留度取所有文本元素的ROUGE-L F1均值：

$$\text{StructValidity} = \frac{1}{|E_{text}|} \sum_{k \in E_{text}} \text{ROUGE-L}_{F1}(\mathbf{c}_k, \mathbf{r}_k)$$

LCS的计算采用 $O(\min(m, n))$ 空间复杂度的动态规划算法，避免在长文本场景下的内存开销。

### 4.3.5 样式丰富度（CSS Richness）

样式丰富度用于衡量候选HTML在视觉表达上的多样性与复杂度。与布局对齐度、文本保留度等“内容正确性”指标不同，该指标关注生成结果是否具备充分的样式设计能力。

具体地，本方法从HTML中的CSS代码中提取两类统计量：

（1）样式长度归一化分数：以CSS字符长度衡量样式信息量，并进行归一化；

（2）属性多样性分数：统计关键样式属性集合（如颜色、阴影、渐变、滤镜、边框、动画、变换等）在候选中的覆盖比例。

设长度归一化分数为 $s_{\text{len}}$，属性多样性分数为 $s_{\text{div}}$，则样式丰富度定义为：

$$\text{CSSRich} = 0.4 \cdot s_{\text{len}} + 0.6 \cdot s_{\text{div}}$$

其中更高的属性多样性权重用于鼓励“有效样式”而非单纯冗长代码。该指标取值范围为 $[0,1]$。

### 4.3.6 空间利用率（Space Utilization）

空间利用率用于衡量页面在1920×1080画布上的有效占用程度，反映候选是否存在大面积留白、布局过于稀疏等问题。该指标由页面注入脚本计算元素占用面积占比并进行归一化，记为：

$$\text{SpaceUtil} \in [0,1]$$

该指标不直接参与偏好对选择的 `score`，而是主要作用于训练阶段的 `reward`，用于引导模型生成更饱满、可读性更高的版式。

### 4.3.7 标签闭合完备度（Tag Closure）

标签闭合完备度衡量HTML结构的语法健壮性。实践中，语言模型生成的长HTML容易出现标签遗漏闭合、嵌套错位等问题，这类问题可能不立即触发运行时报错，但会降低后续渲染稳定性。

本方法采用轻量级启发式检测器，对常见成对标签的开闭匹配情况进行统计，得到完备度分数：

$$\text{TagClosure} \in [0,1]$$

该指标在 `score` 和 `reward` 中均作为辅助项，避免模型通过“视觉投机”获得高分而忽略代码结构质量。

## 4.4 双轨评分体系：Score与Reward的解耦设计

EGPO采用双轨评分体系：`score` 用于偏好对选择（chosen/rejected），`reward` 用于训练信号构造。两者共享同一组执行反馈指标，但优化侧重点不同。

首先，定义DOM覆盖率门控因子：

$$g = \text{DOMCoverage}^{\alpha}, \quad \alpha=1.5$$

其中指数 $\alpha$ 控制覆盖率惩罚强度，默认取1.5以强化“元素缺失”惩罚。

### 4.4.1 用于偏好对选择的Score

若渲染无效（`valid_render=False`），则直接置：

$$\text{Score}=0$$

若渲染有效，则先计算正确性导向质量项：

$$Q = w_l \cdot \text{LayoutAlign} + w_s \cdot \text{StructValidity} + w_c \cdot \text{CSSRich} + w_t \cdot \text{TagClosure}$$

其中默认权重为：

$$w_l=0.45,\; w_s=0.25,\; w_c=0.20,\; w_t=0.10$$

最终用于排序选择的分数为：

$$\text{Score} = g \cdot Q$$

该设计强调布局还原与文本一致性，保证chosen样本在“任务正确性”维度上显著优于rejected样本。

### 4.4.2 用于训练引导的Reward

若渲染无效，则置：

$$\text{Reward}=-1$$

若渲染有效，则计算美学导向项：

$$A = 0.20 \cdot \text{LayoutAlign} + 0.10 \cdot \text{StructValidity} + 0.30 \cdot \text{CSSRich} + 0.30 \cdot \text{SpaceUtil} + 0.10 \cdot \text{TagClosure}$$

并映射到 $[-1,1]$ 区间：

$$\text{Reward} = 2 \cdot (g \cdot A) - 1$$

可以看到，`reward` 相比 `score` 提高了样式丰富度与空间利用率的权重，体现“在保证可执行和对齐的前提下，进一步优化视觉表现力”的训练目标。

## 4.5 偏好对构建策略

对于每个输入幻灯片 $s_i$ 的候选集合 $\{h_i^{(j)}\}_{j=1}^N$，首先计算每个候选的 `score` 与 `reward`。随后按 `score` 降序排序，选择最高分样本作为chosen：

$$h_i^+ = \arg\max_j \text{Score}(h_i^{(j)})$$

rejected样本采用“最差样本策略”（worst）：

$$h_i^- = \arg\min_j \text{Score}(h_i^{(j)})$$

若两者分数差小于阈值 $\epsilon$（默认 $10^{-9}$），则该样本对被跳过，避免引入弱偏好或噪声偏好。最终构造DPO训练三元组：

$$\big(p(s_i),\; h_i^+,\; h_i^-\big)$$

其中 $p(s_i)$ 为输入prompt。

## 4.6 DPO训练目标与优化过程

在得到偏好数据集 $\mathcal{D}=\{(x,y_w,y_l)\}$ 后，采用DPO目标对策略模型 $\pi_\theta$ 进行优化。设参考模型为 $\pi_{\text{ref}}$，则单样本损失定义为：

$$
\mathcal{L}_{\text{DPO}}(\theta)=
-\log \sigma \left(
\beta \left[
\log \frac{\pi_\theta(y_w|x)}{\pi_\theta(y_l|x)}
-\log \frac{\pi_{\text{ref}}(y_w|x)}{\pi_{\text{ref}}(y_l|x)}
\right]
\right)
$$

其中 $\sigma(\cdot)$ 为Sigmoid函数，$\beta$ 为偏好强度系数。本文默认 $\beta=0.1$。该目标本质上鼓励策略模型在相对意义上提升chosen相对rejected的条件概率，并通过参考模型项抑制策略偏移过大。

训练实现采用 `trl.DPOTrainer`，并结合LoRA进行参数高效微调。参考模型由Trainer在PEFT模式下自动处理，无需额外维护独立可训练副本。

## 4.7 关键超参数设置

为保证实验可复现性，本文默认参数如下。

### 4.7.1 候选生成与执行评估参数

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `num_candidates` | 4 | 每个样本生成候选数 $N$ |
| `temperature` | 0.7 | 候选采样温度 |
| `top_p` | 0.9 | nucleus sampling截断阈值 |
| `max_new_tokens` | 4096 | 单候选最大生成长度 |
| `timeout_ms` | 80000 | 单候选Playwright渲染超时 |
| `coverage_gate_exp` | 1.5 | DOM覆盖率门控指数 $\alpha$ |
| `epsilon` | $10^{-9}$ | 偏好对最小分差阈值 |
| `rejected_strategy` | `worst` | rejected选择策略 |

### 4.7.2 Score权重参数

| 参数 | 默认值 |
|---|---:|
| `w_layout` | 0.45 |
| `w_struct` | 0.25 |
| `w_css` | 0.20 |
| `w_closure` | 0.10 |

### 4.7.3 DPO与LoRA训练参数

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `num_epochs` | 3 | 训练轮数 |
| `lr` | 5e-6 | 学习率 |
| `batch_size` | 2 | 单卡batch size |
| `grad_accum` | 8 | 梯度累积步数 |
| `max_length` | 4096 | 序列最大长度 |
| `max_prompt_length` | 2048 | prompt最大长度 |
| `beta` | 0.1 | DPO偏好强度系数 |
| `warmup_ratio` | 0.1 | 学习率预热比例 |
| `lr_scheduler_type` | `cosine` | 学习率调度策略 |
| `lora_r` | 16 | LoRA秩 |
| `lora_alpha` | 32 | LoRA缩放系数 |
| `lora_dropout` | 0.05 | LoRA dropout |
| `lora_target_modules` | `q_proj,v_proj,k_proj,o_proj,gate_proj,up_proj,down_proj` | LoRA注入模块 |

上述参数在默认设置下能够在“可执行性、结构还原度与视觉表现力”之间取得稳定平衡。后续章节实验部分将进一步通过消融实验验证各参数及各指标权重对最终性能的影响。
