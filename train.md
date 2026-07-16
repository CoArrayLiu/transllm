# 四数据集两阶段训练（A100 80GB）

在项目根目录执行：

~~~bash
conda activate transllm
~~~

当前入口只支持单 GPU；代码会拒绝 DDP，避免四数据集的同源 batch 被分布式 sampler 打乱。

## A100 实测

本机为 A100-SXM4-80GB，BF16、`model_max_length=1024`。短基准用于选型，吞吐受首步预热影响：

| 阶段 | batch | checkpointing | allocated | reserved | 吞吐 |
|---|---:|---:|---:|---:|---:|
| Stage 1 | 2 | true | 19.8 GiB | 21.3 GiB | 3.37 samples/s |
| Stage 1 | 4 | false | 32.6 GiB | 36.0 GiB | 6.21 samples/s |
| Stage 1 | 8 | false | 48.4 GiB | 52.1 GiB | 7.43 samples/s |
| Stage 1 | 12 | false | 63.3 GiB | 66.1 GiB | 7.70 samples/s |
| Stage 2 | 32 | false | 17.6 GiB | 18.6 GiB | 14.46 samples/s |
| Stage 2 | 64 | false | 21.4 GiB | 23.6 GiB | 16.77 samples/s |

Router batch=64 时采样到 GPU 利用率 100%、约 400W。正式建议：

- Stage 1：batch=12、accumulation=1、关闭 gradient checkpointing；若同机有其他显存占用，退到 batch=8，不建议直接上 batch=16。
- Stage 2：batch=64、accumulation=1。Router 阶段已跳过无梯度价值的 128K 词表 logits/交叉熵。
- `model_max_length=1024` 已过四数据集烟测。代码现在禁止静默截断，超长样本会报告实际 token 数并中止。
- workers=2 已能运行，但短测不比 workers=0 快，正式基线使用 workers=0；要调整时先做至少 100 steps 对照。

全局 batch 为 `per_device_train_batch_size × gradient_accumulation_steps`。增大 accumulation 不能提高单步 GPU 利用率，改变全局 batch 也不会自动缩放学习率。

## Stage 1：LLM + 预测头

~~~bash
CUDA_VISIBLE_DEVICES=0 python -m transllm.train.train_learning_prompt_5dataset \
  --training_stage llm \
  --output_dir ./checkpoints/transllm_4dataset/stage1_llm \
  --num_train_epochs 1 \
  --per_device_train_batch_size 12 \
  --gradient_accumulation_steps 1 \
  --model_max_length 1024 \
  --gradient_checkpointing false \
  --bf16 true \
  --fp16 false \
  --bits 16 \
  --dataloader_num_workers 0 \
  --learning_rate 1e-4 \
  --max_grad_norm 1.0 \
  --save_strategy steps \
  --save_steps 5000 \
  --save_total_limit 2
~~~

Stage 1 结束后确认存在 `./checkpoints/transllm_4dataset/stage1_llm/full_model`。不要在 Stage 1 尚未收敛时启动 Router，因为 Router reward 直接依赖 Stage 1 的预测误差。

## Stage 2：Router

~~~bash
CUDA_VISIBLE_DEVICES=0 python -m transllm.train.train_learning_prompt_5dataset \
  --training_stage router \
  --model_name_or_path ./checkpoints/transllm_4dataset/stage1_llm/full_model \
  --output_dir ./checkpoints/transllm_4dataset/stage2_router \
  --num_train_epochs 1 \
  --per_device_train_batch_size 64 \
  --gradient_accumulation_steps 1 \
  --model_max_length 1024 \
  --gradient_checkpointing false \
  --bf16 true \
  --fp16 false \
  --bits 16 \
  --dataloader_num_workers 0 \
  --learning_rate 1e-4 \
  --max_grad_norm 1.0 \
  --save_strategy steps \
  --save_steps 5000 \
  --save_total_limit 2
~~~

Router 参数保持 FP32，Llama 主体保持 BF16。Router 总 loss 包含冻结预测分支的任务 loss，因此不同数据集量纲差异较大；同时观察 loss 是否有限、`grad_norm`、下游验证指标及参数更新。

## 长训练前四步预跑

四步依次覆盖 SD、pems08、SZ、urbanev。烟测关闭 warmup，并跳过最终约 16GB 的完整模型保存：

~~~bash
CUDA_VISIBLE_DEVICES=0 python -m transllm.train.train_learning_prompt_5dataset \
  --training_stage llm \
  --output_dir /tmp/transllm_preflight \
  --max_steps 4 \
  --warmup_ratio 0 \
  --per_device_train_batch_size 12 \
  --gradient_accumulation_steps 1 \
  --model_max_length 1024 \
  --gradient_checkpointing false \
  --dataloader_num_workers 0 \
  --report_to none \
  --save_strategy no \
  --skip_final_save true
~~~

`--skip_final_save true` 只允许与正数 `--max_steps` 一起使用。一步烟测也必须加 `--warmup_ratio 0`，否则唯一优化步学习率为零，入口会拒绝启动。

Router 预跑只需将阶段改为 router，加上 Stage 1 的 `--model_name_or_path`，并把 batch 改为 64；其他烟测参数保持一致。

## 检查点间隔与恢复

`save_steps` 按 optimizer step 计数：

~~~text
每次检查点覆盖样本数 =
  save_steps × per_device_train_batch_size × gradient_accumulation_steps
~~~

Stage 1 推荐配置下，`save_steps=5000` 约覆盖 60000 个样本。检查点含 adapter、预测头、projector、lm_head、optimizer、scheduler 和 RNG；`save_total_limit=2` 保留最近两个。

恢复时保持原 batch、梯度累积、学习率、scheduler、阶段和输出目录：

~~~bash
CUDA_VISIBLE_DEVICES=0 python -m transllm.train.train_learning_prompt_5dataset \
  --training_stage llm \
  --output_dir ./checkpoints/transllm_4dataset/stage1_llm \
  --resume_checkpoint ./checkpoints/transllm_4dataset/stage1_llm/checkpoint-5000 \
  --per_device_train_batch_size 12 \
  --gradient_accumulation_steps 1 \
  --model_max_length 1024 \
  --gradient_checkpointing false \
  --learning_rate 1e-4 \
  --save_steps 5000 \
  --save_total_limit 2

CUDA_VISIBLE_DEVICES=0 python -m transllm.train.train_learning_prompt_5dataset \
  --training_stage llm \
  --model_name_or_path ./checkpoints/llama3-8b \
  --output_dir ./checkpoints/transllm_4dataset/stage1_llm \
  --resume_checkpoint ./checkpoints/transllm_4dataset/stage1_llm/checkpoint-10000 \
  --num_train_epochs 1 \
  --per_device_train_batch_size 12 \
  --gradient_accumulation_steps 1 \
  --model_max_length 1024 \
  --gradient_checkpointing false \
  --bf16 true \
  --fp16 false \
  --bits 16 \
  --dataloader_num_workers 0 \
  --learning_rate 1e-4 \
  --lr_scheduler_type cosine \
  --warmup_ratio 0.03 \
  --max_grad_norm 1.0 \
  --seed 42 \
  --save_strategy steps \
  --save_steps 5000 \
  --save_total_limit 2
~~~

## 运行时检查与系统项

另开终端运行 `watch -n 1 nvidia-smi`。入口会打印 `training_resource_config`，结束时打印 `training_gpu_stats`，应核对 batch、accumulation、dtype、长度、workers 和峰值显存。

PyG 扩展已从错误的 `pt25cu124` 更新为匹配 `torch 2.6.0+cu124` 的 `pt26cu124`，pyg-lib/scatter/sparse 原生扩展均已验证加载。

当前 Linux kernel 5.4 低于 Accelerate 建议的 5.5。烟测正常，但这是项目代码无法消除的潜在长任务 hang 风险；能维护系统时优先升级到 5.5 或更新的 LTS 内核。


## 使用中间 checkpoint 评估

评估入口支持 `--checkpoint` 直接重建 Stage 1 的 LoRA checkpoint，也支持用 `--num-samples` 从 `--start_id` 开始限制样本数。例如先评估 pems08 的一个完整时间窗（170 个节点样本）：

~~~bash
CUDA_VISIBLE_DEVICES=0 python -m transllm.test.run_transllm \
  --checkpoint ./checkpoints/transllm_4dataset/stage1_llm/checkpoint-10000 \
  --base-model ./checkpoints/llama3-8b \
  --fixed-prompt-index 0 \
  --prompting_file ./data/prompt_data/pems08_test.json \
  --st_data_path ./data/prompt_data/pems08_test_pkl.pkl \
  --output_res_path ./result_checkpoint/checkpoint-10000/pems08 \
  --start_id 0 \
  --num-samples 170 \
  --max_new_tokens 256 \
  --num_gpus 1 
~~~

`--num-samples N` 与 `--end_id` 二选一。为使区域指标覆盖完整时间窗，`start_id` 和样本数最好按节点数对齐：SD 673、SZ 247、pems08 170、urbanev 275；快速功能烟测可以只跑少量样本，但不能把这种局部结果当作完整数据集指标。

当前已经启动的 Stage 1 进程加载的是修改前的 checkpoint 保存逻辑，因此该次运行产生的 `checkpoint-N` 没有冻结 Router 快照，评估时必须指定同一个 `--fixed-prompt-index` 才能公平比较。之后重新启动的新训练会在 checkpoint 中保存 Router，此时可去掉该参数；Stage 1 的 Router 本来就是冻结的，为减少随机 Router 对 checkpoint 比较的干扰，仍推荐统一使用固定提示。最终的 `full_model` 是完整模型，可使用 `--model-name` 直接评估。

生成结果后计算指标：

~~~bash
python -m metric_calculation.result_test \
  --folder_path ./result_checkpoint/checkpoint-10000/pems08 \
  --dataset pems08
~~~

## 数据切分说明

四数据集已经按时间分离 supervised 训练段和 test 段，并非从同一批样本中随机拆分；但当前没有独立 validation 集。具体原始时间范围为：SD 使用最后 9 天训练、倒数第 43 天到第 35 天测试；pems08/SZ 使用倒数第 28 天到第 17 天训练、最后 8 天测试；urbanev 使用倒数第 50 天到第 17 天训练、最后 10 天测试。窗口构造还需要一周历史，因此测试 JSON 中分别得到 265、265、265、49 个预测时间窗。 SD 的测试段早于训练段，虽然没有样本重叠，但不是标准的向未来时间外推；若论文指标要求严格 temporal holdout，应重新划分 SD。

不要反复根据 test 指标挑选最佳 checkpoint，否则 test 会事实上变成 validation，最终指标会偏乐观。正式选模型应另划时间验证段，test 只在方案确定后运行一次。测试数据现在强制复用 supervised 段的 mean/std，避免测试分布泄漏；已有 prompt 文件是在修复前生成的，需要在当前长训练结束后重新执行 `./scripts/generate_prompt_data.sh`，再用于最终评估。这个修复不影响当前已在内存中训练的模型，也不要求重训。
