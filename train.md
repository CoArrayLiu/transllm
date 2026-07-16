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
~~~

## 运行时检查与系统项

另开终端运行 `watch -n 1 nvidia-smi`。入口会打印 `training_resource_config`，结束时打印 `training_gpu_stats`，应核对 batch、accumulation、dtype、长度、workers 和峰值显存。

PyG 扩展已从错误的 `pt25cu124` 更新为匹配 `torch 2.6.0+cu124` 的 `pt26cu124`，pyg-lib/scatter/sparse 原生扩展均已验证加载。

当前 Linux kernel 5.4 低于 Accelerate 建议的 5.5。烟测正常，但这是项目代码无法消除的潜在长任务 hang 风险；能维护系统时优先升级到 5.5 或更新的 LTS 内核。
