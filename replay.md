Action:
配置数据集

修改了包导入问题   和生成使用的prompt文件夹名称不一致 问题

增加了修改数据集文件夹名称的脚本

增加了生成  指令 和 缓存文件 的 bash脚本  ./scripts

### commit fix bugs

```bash
./scripts/normalize_dataset_paths.sh
./scripts/generate_cache_matrices.sh
./scripts/generate_prompt_data.sh
```

下载LLM 配置config.json
```
python -m pip install modelscope

modelscope download \
  --model LLM-Research/Meta-Llama-3.1-8B-Instruct \
  --local_dir checkpoints/llama3-8b \
  --exclude 'original/*'

cp  backup/config.json  checkpoints/llama3-8b/config.json
```

### commit  configuration

修改了SH数据集相关代码

### commit remove SH related code

修复了训练推理的一些bug

### commit fix some train-eval bugs

修复test和train数据集std mean 不一致导致的评估不一致的bug   ./scripts/generate_prompt_data.sh

断点续训会不会重新遍历数据集，断点续训指令

检查点test与eval

增加stage1检查点也会保存router权重

### commit fix some checkpoints eval bugs

检测到<ST-START>后继续输出12个<ST-PATCH>后就直接停止，加速预测

修复评估：允许缺少 <ST_start> 时提取第一段连续 12 个 patch

stage1评估不再使用固定prompt

训练过程对各数据集回归 loss 做尺度归一化，并单独记录 language_loss、regression_loss。

增加四数据集统一快速评测脚本

### commit improve train and eval detailss



Q:

可以只用当前四个数据集跑，但需要做一版“4-dataset”适配；仓库当前不能原样运行。

当前四个数据集：

- `SD`、`PEMS08`：交通流预测
- `SZ`、`UrbanEV`：充电需求预测
- 缺少 `SH` 后，不能复现车辆调度任务及对应指标

主要阻塞点：

- 训练集类会无条件读取 SH 的 JSON、PKL 和上海网格文件，缺失就立即报错：[train_st_learning_prompt_5dataset.py (line 446)](/home/liu/code/TransLLM/transllm/train/train_st_learning_prompt_5dataset.py:446)
- 邻接矩阵加载也写死了 `shanghai`：[train_st_learning_prompt_5dataset.py (line 230)](/home/liu/code/TransLLM/transllm/train/train_st_learning_prompt_5dataset.py:230)
- 代码期望 `data/st_data/sd`、`shenzhen`，实际目录是大写 `SD`、`SZ`。
- 四个数据集都缺少 `cached_dist_matrix.npy`。
- 尚未生成 `data/prompt_data/*.json` 和 `*.pkl`，训练入口默认直接读取这些文件：[train_learning_prompt_5dataset.py (line 20)](/home/liu/code/TransLLM/transllm/train/train_learning_prompt_5dataset.py:20)
- 缺少基础 Llama checkpoint；当前只有 `st_encoder.pt`。
- 有一个明显错误导入：`TransLLM.transllm...`，实际应从小写 `transllm` 导入：[train_st_learning_prompt_5dataset.py (line 36)](/home/liu/code/TransLLM/transllm/train/train_st_learning_prompt_5dataset.py:36)

推荐复现路线：

1. 使用现有 `st_encoder.pt`，不重新预训练 ST-Encoder。
2. 删除/条件化所有 SH 数据加载和 batch 调度。
3. 修正数据目录、导入路径和邻接矩阵缓存生成。
4. 为四个数据集生成 instruction JSON/PKL。
5. 用 4-bit LoRA 训练，分别评测四个回归数据集。

需要注意：现有 ST-Encoder 权重是作者在五个数据集（包含 SH）上预训练的，[README (line 181)](/home/liu/code/TransLLM/readme.md:181)。因此使用它不需要 SH 原始数据，但严格来说不算“完全只用四个数据集从零训练”。如果要求严格四数据集重训 ST-Encoder，当前的 [pretrain_Enc.py (line 10)](/home/liu/code/TransLLM/pretrain_encoder/pretrain_Enc.py:10) 还依赖仓库中不存在的 `src` 模块，需额外重构。

所以最终判断是：**能复现四个预测数据集的训练和评测，不能复现 SH 调度任务；大约需要一轮中等规模代码适配。** 我这轮只做了检查，没有修改现有文件。