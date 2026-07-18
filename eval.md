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


Train Record:
paper 16*3600/1.5=38400
r 35000  pems08   mae 12.15  RMSE 16.03  MAPE 35%
r 40000  pems08   mae 11.71  RMSE 15.04  MAPE 44%
r 40000  pems08   mae 10.55  RMSE 13.97  MAPE 38%
r 50000 Average Horizon, MAE: 10.22, RMSE: 13.65, MAPE: 27.4503%
r 70000 Average Horizon, MAE: 8.75, RMSE: 11.99, MAPE: 19.1478%s

CUDA_VISIBLE_DEVICES=0 python -m transllm.test.run_transllm \
  --checkpoint ./checkpoints/transllm_4dataset/stage1_llm/checkpoint-65000 \
  --base-model ./checkpoints/llama3-8b \
  --prompting_file ./data/prompt_data/pems08_test.json \
  --st_data_path ./data/prompt_data/pems08_test_pkl.pkl \
  --output_res_path ./result_checkpoint/checkpoint-65000/pems08_12windows \
  --start_id 0 \
  --num-samples 340 \
  --max_new_tokens 256 \
  --num_gpus 1


python -m metric_calculation.result_test \
  --folder_path ./result_checkpoint/checkpoint-65000/pems08_12windows \
  --dataset pems08


scripts/quick_eval_4datasets.sh \
  ./checkpoints/transllm_4dataset/stage1_llm/checkpoint-50000 \
  170 \
  ./result_checkpoint/checkpoint-50000/quick_test