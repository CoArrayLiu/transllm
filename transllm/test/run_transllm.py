import argparse
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    StoppingCriteria,
    StoppingCriteriaList,
)
import torch
import torch.nn  as nn
import sys
import re
import copy
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(os.path.split(curPath)[0])[0]
print(curPath, rootPath)
sys.path.append(rootPath)
from transllm.train.train_st_learning_prompt_5dataset import get_config, load_adj
from transllm.conversation import conv_templates, SeparatorStyle
from transllm.utils import disable_torch_init
from transllm.model import *
from transllm.model.utils import KeywordsStoppingCriteria
import json
from transllm.model.STLlama_learning_prompt_5dataset import STLlamaForCausalLM
import numpy as np
from tqdm import tqdm
import json
import os.path as osp
import pickle

# import ray

DEFAULT_STHIS_TOKEN = "<ST_EMB>"
DEFAULT_STPRE_TOKEN = "<ST_PRE>"
DEFAULT_ST_PATCH_TOKEN = "<ST_patch>"
DEFAULT_ST_START_TOKEN = "<ST_start>"
DEFAULT_ST_END_TOKEN = "<ST_end>"


class StopAfterSTPrediction(StoppingCriteria):
    """Stop after all prediction-patch hidden states have been computed.

    Generation advances once beyond the final patch because a sampled token's
    hidden state is produced by the following decoder forward pass.
    """

    def __init__(self, start_token_id, patch_token_id, patch_count=12):
        self.start_token_id = int(start_token_id)
        self.patch_token_id = int(patch_token_id)
        self.patch_count = int(patch_count)

    def __call__(self, input_ids, scores, **kwargs):
        # Keep one look-ahead token so the final patch hidden state has been
        # computed by the decoder before generation stops.
        completed_length = self.patch_count + 1
        if input_ids.shape[1] < completed_length:
            return False

        tail = input_ids[:, -completed_length:]
        contains_all_patches = tail[:, :-1].eq(self.patch_token_id).all(dim=1)
        return bool(contains_all_patches.all().item())


def find_first_patch_run(token_ids, patch_token_id, patch_count=12):
    """Return the first relative index of consecutive prediction patches."""
    if token_ids.numel() < patch_count:
        return None
    patch_mask = token_ids.eq(int(patch_token_id))
    for start in range(token_ids.numel() - patch_count + 1):
        if bool(patch_mask[start:start + patch_count].all().item()):
            return start
    return None


def parse_st_id(instruction_id):
    """Extract region bounds and the cached sample index from a prompt ID."""
    match = re.fullmatch(
        r"train_(?P<dataset>.+)_region_(?P<region_start>\d+)_"
        r"(?P<region_end>\d+)_len_(?P<sample_index>\d+)",
        instruction_id,
    )
    if match is None:
        raise ValueError(
            f"Invalid instruction id {instruction_id!r}; expected "
            "'train_<dataset>_region_<start>_<end>_len_<sample_index>'"
        )
    return (
        int(match.group("region_start")),
        int(match.group("region_end")),
        int(match.group("sample_index")),
    )



def load_st(idx, instruct_item, st_data_all):

    sources = instruct_item

    region_start, region_end, i4data_all = parse_st_id(sources["id"])

    st_data_x = torch.Tensor(st_data_all[i4data_all]['data_x'])
    st_data_xd = torch.Tensor(st_data_all[i4data_all]['data_x_1d'])
    st_data_xw = torch.Tensor(st_data_all[i4data_all]['data_x_1w'])
    st_data_y = torch.Tensor(st_data_all[i4data_all]['data_y'])

    mean = torch.tensor(st_data_all[i4data_all]['mean'])
    std = torch.tensor(st_data_all[i4data_all]['std'])

    cur_token_len = 1

    return {
        'st_data_x': st_data_x,
        'st_data_xd': st_data_xd,
        'st_data_xw': st_data_xw,
        'st_data_y': st_data_y,
        'mean' : mean,
        'std' : std,
        'region_start': region_start,
        'region_end': region_end,
        'st_token_len': cur_token_len
    }


def load_prompting_file(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data




def run_eval(args, num_gpus):
    if num_gpus != 1:
        raise ValueError(
            "This evaluator runs one process on one GPU; launch separate processes "
            "with disjoint start_id/end_id ranges for multi-GPU evaluation"
        )
    prompt_file = load_prompting_file(args.prompting_file)
    total_samples = len(prompt_file)
    if args.start_id < 0 or args.start_id >= total_samples:
        raise ValueError(
            f"start_id={args.start_id} is outside [0, {total_samples})"
        )
    if args.end_id <= args.start_id or args.end_id > total_samples:
        raise ValueError(
            f"end_id={args.end_id} must be in "
            f"({args.start_id}, {total_samples}]"
        )
    selected_prompts = prompt_file[args.start_id:args.end_id]
    print(
        "evaluation_sample_range:",
        {
            "start_id": args.start_id,
            "end_id": args.end_id,
            "num_samples": len(selected_prompts),
            "total_samples": total_samples,
        },
    )
    os.makedirs(args.output_res_path, exist_ok=True)
    eval_model(
        args,
        selected_prompts,
        args.start_id,
        args.end_id,
    )

def replace_prompt_sd(routing_info: torch.Tensor, original_sentence: str,st_data_xh_tmp,st_data_xd_tmp,st_data_xw_tmp) -> str:
        traffic_values, history_time, future_time = extract_info(original_sentence=original_sentence)
        st_data_xh = st_data_xh_tmp[:,0,0].int()
        st_data_xd = st_data_xd_tmp[:,0,0].int()
        st_data_xw = st_data_xw_tmp[:,0,0].int()
        slot_0_history_flow = [
            f"Using traffic flow recorded one hour ago {st_data_xh.tolist()}, we examine recent short-term variations to identify immediate trends.",
            
            f"To uncover daily temporal patterns, we utilize both the traffic from one hour ago {st_data_xh.tolist()} and from the same time yesterday {st_data_xd.tolist()}, enabling comparison across consecutive days.",
            
            f"Recognizing weekly rhythms, we analyze flow data from one hour ago {st_data_xh.tolist()} alongside that from the same time last week {st_data_xw.tolist()}, aiming to detect recurring weekly behaviors.",
            
            f"By jointly considering traffic observations from one hour ago {st_data_xh.tolist()}, the same time yesterday {st_data_xd.tolist()}, and the same time last week {st_data_xw.tolist()}, we explore both short-term fluctuations and long-term temporal dependencies."
        ]

        slot_1_time_context = [
            f"Based on these inputs, the task is to forecast traffic for the next hour starting from {future_time}, with each step representing a 5-minute interval to capture short-term shifts.",

            f"Grounded on these historical cues, we now look ahead from {future_time}, forecasting 12 future values at 5-minute intervals to reveal potential periodic trends.",

            f"Given the temporal scope of the input, we proceed to estimate future traffic patterns starting at {future_time}, carefully adjusting for time-of-day effects such as rush hours or quiet periods.",

            f"With this historical context in mind, the model will generate a 12-step forecast beginning at {future_time}, incorporating 5-minute resolution to synthesize both rapid changes and gradual trends."
        ]


        slot_2_task_description = [
            "To begin the forecasting process, we utilize a pretrained spatio-temporal encoder to capture dependencies within the 12-step prediction horizon: <ST_EMB>.",
            
            "In order to better understand the early dynamics of traffic flow, we incorporate 12 spatio-temporal embeddings that correspond to future time intervals: <ST_EMB>.",
            
            "Specifically, a sequence of 12 step tokens is employed to represent the spatio-temporal context within the 12-step prediction window: <ST_EMB>.",
            
            "To provide comprehensive temporal context, the spatio-temporal encoder embeds all 12 predicted intervals with dedicated representations, focusing on early-stage forecasting: <ST_EMB>."
        ]

        slot_3_answer_format = [  
            "Please reason step-by-step through both temporal patterns and spatial influences. After that, generate the predicted traffic volume for the next 12 time intervals using the token <ST_PRE>.",
            
            "First, analyze recent flow trends and any irregular fluctuations you observe. Then, estimate future traffic volumes and present your answer using <ST_PRE>.",
            
            "To guide your prediction, draw from common traffic behaviors—such as rush hour surges or off-peak periods—and conclude your response with the 12-step forecast in <ST_PRE>.",
            
            "Consider potential causes of congestion or unexpected volume shifts. Based on this reasoning, produce a stable and well-grounded 12-step forecast using <ST_PRE>."
        ]



        slots = [
            slot_0_history_flow,
            slot_1_time_context,
            slot_2_task_description,
            slot_3_answer_format
        ]
         
        prompt_parts = [slots[i][routing_info[i]] for i in range(4)]
        new_sentence = " ".join(prompt_parts)
        return new_sentence
def replace_prompt_sz(routing_info: torch.Tensor, original_sentence: str,
                      st_data_xh_tmp, st_data_xd_tmp, st_data_xw_tmp) -> str:
        traffic_values, history_time, future_time = extract_info(original_sentence=original_sentence)
        st_data_xh = st_data_xh_tmp[:, 0, 0].int()
        st_data_xd = st_data_xd_tmp[:, 0, 0].int()
        st_data_xw = st_data_xw_tmp[:, 0, 0].int()

        slot_0_history_flow = [
            f"Using charging load recorded one hour ago {st_data_xh.tolist()}, we investigate short-term dynamics to capture immediate fluctuations.",
            f"To explore daily charging behavior, we compare the load from one hour ago {st_data_xh.tolist()} and the same time yesterday {st_data_xd.tolist()}, helping detect day-over-day shifts.",
            f"To identify weekly demand patterns, we reference charging data from one hour ago {st_data_xh.tolist()} and the same time last week {st_data_xw.tolist()}, examining recurring temporal signals.",
            f"By jointly analyzing charging records from one hour ago {st_data_xh.tolist()}, yesterday {st_data_xd.tolist()}, and last week {st_data_xw.tolist()}, we model both near-term and long-range dependencies."
        ]

        slot_1_time_context = [
            f"Based on these temporal references, the task is to forecast charging load for the next hour starting from {future_time}, with 5-minute intervals capturing fine-grained changes.",
            f"Leveraging these time-aligned insights, we aim to predict the 12-step charging demand starting at {future_time}, revealing emerging patterns.",
            f"Considering the timing context, we now estimate near-future charging needs from {future_time} forward, incorporating common time-of-day usage trends.",
            f"With this multi-scale temporal context, the forecast begins at {future_time}, enabling the model to learn both abrupt shifts and smoother demand transitions."
        ]

        slot_2_task_description = [
            "To begin the prediction process, we utilize a pretrained spatio-temporal encoder that captures dependencies from the 12 future steps: <ST_EMB>.",
            "For better modeling of early-stage charging dynamics, 12patio-temporal embeddings are used to represent the future time intervals: <ST_EMB>.",
            "The encoder uses a sequence of 12 step tokens to represent spatio-temporal context over the prediction steps: <ST_EMB>.",
            "A complete spatio-temporal embedding of all 12 predicted intervals is used, focusing on the early part of the forecast: <ST_EMB>."
        ]
        
        slot_3_answer_format = [
            "Please reason step-by-step through both temporal rhythms and spatial correlations. Then generate the 12-step charging load forecast using <ST_PRE>.",
            "Begin by analyzing trends and irregularities in recent data. Based on your reasoning, generate your forecast using <ST_PRE>.",
            "Rely on knowledge of typical EV charging demand cycles—like peak periods—to guide your reasoning, then provide the result via <ST_PRE>.",
            "Consider both recurring patterns and unexpected shifts in demand. Then produce a robust 12-step forecast using <ST_PRE>."
        ]

        slots = [
            slot_0_history_flow,
            slot_1_time_context,
            slot_2_task_description,
            slot_3_answer_format
        ]

        prompt_parts = [slots[i][routing_info[i]] for i in range(4)]
        new_sentence = " ".join(prompt_parts)
        return new_sentence


def extract_info(original_sentence: str):
    traffic_match = re.search(r"traffic flow values are \[([^\]]+)\]", original_sentence)
    traffic_values = traffic_match.group(1).strip() if traffic_match else "unknown"

    history_time_match = re.search(
        r"The recording time of the historical data is '([^']*?)(?= with data points recorded)", original_sentence)
    history_time = history_time_match.group(1).strip().rstrip(',') if history_time_match else "unknown"

    future_time_match = re.search(
        r"the next \d+ time steps during the time period of '([^']*?)(?= with data points recorded)", original_sentence)
    future_time = future_time_match.group(1).strip().rstrip(',') if future_time_match else "unknown"

    return traffic_values, history_time, future_time

# @ray.remote(num_gpus=1)
def load_evaluation_model(args):
    """Load either a merged model or a mathematically complete Stage 1 checkpoint."""
    if args.checkpoint is None:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        model = STLlamaForCausalLM.from_pretrained(
            args.model_name,
            num_prompts=4,
            num_slots=4,
            torch_dtype=torch.bfloat16,
            use_cache=True,
            device_map=None,
            low_cpu_mem_usage=False,
        )
        model.set_tokenizer(tokenizer)
        return tokenizer, model

    checkpoint_path = osp.abspath(args.checkpoint)
    required_files = (
        "adapter_config.json",
        "adapter_model.safetensors",
        "non_lora_trainables.bin",
        "tokenizer_config.json",
    )
    missing_files = [
        filename
        for filename in required_files
        if not osp.isfile(osp.join(checkpoint_path, filename))
    ]
    if missing_files:
        raise FileNotFoundError(
            f"Incomplete Stage 1 checkpoint {checkpoint_path}: missing {missing_files}"
        )

    with open(osp.join(checkpoint_path, "adapter_config.json"), "r") as handle:
        adapter_config = json.load(handle)
    base_model_path = args.base_model or adapter_config.get(
        "base_model_name_or_path"
    )
    if not base_model_path:
        raise ValueError(
            "The checkpoint does not identify its base model; pass --base-model"
        )

    non_lora_path = osp.join(checkpoint_path, "non_lora_trainables.bin")
    non_lora_state = torch.load(
        non_lora_path,
        map_location="cpu",
        weights_only=True,
    )
    required_router_prefixes = (
        "prompt_router_sd.",
        "prompt_router_pems08.",
        "prompt_router_sz.",
        "prompt_router_urbanev.",
    )
    missing_routers = [
        prefix
        for prefix in required_router_prefixes
        if not any(prefix in key for key in non_lora_state)
    ]
    if missing_routers:
        if args.fixed_prompt_index is None:
            raise RuntimeError(
                f"Checkpoint {checkpoint_path} predates complete router saving and "
                f"cannot reproduce its exact evaluation behavior; missing "
                f"{missing_routers}. Pass --fixed-prompt-index 0 to compare "
                "Stage 1 prediction weights with a fixed prompt policy."
            )
        print(
            "Checkpoint has no frozen router snapshot; using fixed prompt index "
            f"{args.fixed_prompt_index} for Stage 1 comparison"
        )

    checkpoint_tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    if checkpoint_tokenizer.pad_token is None:
        raise ValueError("Checkpoint tokenizer has no pad token")
    tokenizer.pad_token = checkpoint_tokenizer.pad_token
    model = STLlamaForCausalLM.from_pretrained(
        base_model_path,
        num_prompts=4,
        num_slots=4,
        torch_dtype=torch.bfloat16,
        use_cache=True,
        device_map=None,
        low_cpu_mem_usage=False,
    )
    model.set_tokenizer(tokenizer)
    # The base Llama config does not create the ST projector. Training creates
    # it explicitly after loading the base model, so checkpoint reconstruction
    # must perform the same initialization before loading non-LoRA weights.
    model.get_model().initialize_st_modules(
        st_tower="ST_Encoder",
        st_select_layer=-2,
        pretrain_st_mlp_adapter=None,
        fsdp=None,
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_st_start_end = True
    model.initialize_st_tokenizer(
        use_st_start_end=True,
        tokenizer=tokenizer,
        device="cpu",
    )
    required_st_tokens = (
        DEFAULT_ST_PATCH_TOKEN,
        DEFAULT_ST_START_TOKEN,
        DEFAULT_ST_END_TOKEN,
    )
    if len(tokenizer) != len(checkpoint_tokenizer) or any(
        tokenizer.convert_tokens_to_ids(token)
        != checkpoint_tokenizer.convert_tokens_to_ids(token)
        for token in required_st_tokens
    ):
        raise RuntimeError(
            "Base/checkpoint tokenizer ST token layouts do not match"
        )
    tokenizer = checkpoint_tokenizer
    model.set_tokenizer(tokenizer)

    from peft import PeftModel
    model = PeftModel.from_pretrained(
        model,
        checkpoint_path,
        is_trainable=False,
    )
    incompatible = model.load_state_dict(non_lora_state, strict=False)
    if incompatible.unexpected_keys:
        raise RuntimeError(
            "Unexpected non-LoRA checkpoint keys: "
            f"{incompatible.unexpected_keys}"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("Checkpoint evaluation requires a CUDA GPU")
    # Merging an 8B LoRA model on CPU is unnecessarily slow. Evaluation is a
    # CUDA-only path, so move the reconstructed PEFT model before merging.
    model = model.to("cuda")
    model = model.merge_and_unload()
    # The trained projector is FP32, while generation supplies BF16 ST
    # features. Use the model compute dtype for inference Linear operations.
    model.get_model().st_projector.to(dtype=torch.bfloat16)
    model.get_model().st_projector_sh.to(dtype=torch.bfloat16)
    model.set_tokenizer(tokenizer)
    model.config.use_cache = True
    print(
        f"Loaded Stage 1 checkpoint {checkpoint_path} with base model "
        f"{base_model_path}"
    )
    return tokenizer, model


def select_routing_info(
    args,
    router,
    st_tower,
    st_inputs,
    sp_matrix,
    se_matrix,
    data_type,
    node_feature,
    region_start,
    region_end,
):
    if args.fixed_prompt_index is not None:
        return torch.full(
            (4,),
            args.fixed_prompt_index,
            dtype=torch.long,
            device=st_inputs.device,
        )
    _, node_embedding = st_tower(
        st_inputs,
        sp_matrix,
        se_matrix,
        data_type,
        node_feature,
    )
    selected = node_embedding[:, :, region_start:region_end, :]
    router_dtype = next(router.parameters()).dtype
    return router.select_prompts(
        selected.reshape(selected.shape[0], -1).to(router_dtype),
        track_episode=False,
        deterministic=True,
    ).squeeze(0)


@torch.inference_mode()
def eval_model(
    args,
    prompt_file,
    start_idx,
    end_idx,
    tokenizer=None,
    model=None,
    graphs=None,
):
    
    disable_torch_init()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    if (tokenizer is None) != (model is None):
        raise ValueError("tokenizer and model must be supplied together")
    if tokenizer is None:
        print('start loading')
        tokenizer, model = load_evaluation_model(args)
    else:
        print('reusing loaded evaluation model')
    model = model.to("cuda")
    model.get_st_tower().to(device="cuda", dtype=torch.float32)
    model.eval()
    print('finish loading')
    if graphs is None:
        args1 = get_config()
        args1.bs = 1
        prompt_datasets = {
            item["id"].split("_")[1]
            for item in prompt_file
        }
        zero_shot_datasets = prompt_datasets & {"pems03", "pems04"}
        graphs = load_adj(args1, additional_datasets=zero_shot_datasets)
   
    use_st_start_end = getattr(model.config, "use_st_start_end", True)
    if not use_st_start_end:
        raise ValueError("The four-dataset checkpoint must use ST start/end tokens")
    required_tokens = [
        DEFAULT_ST_PATCH_TOKEN,
        DEFAULT_ST_START_TOKEN,
        DEFAULT_ST_END_TOKEN,
    ]
    required_ids = tokenizer.convert_tokens_to_ids(required_tokens)
    embedding_count = model.get_input_embeddings().num_embeddings
    invalid_tokens = [
        token
        for token, token_id in zip(required_tokens, required_ids)
        if token_id is None
        or token_id == tokenizer.unk_token_id
        or token_id >= embedding_count
    ]
    if invalid_tokens:
        raise ValueError(
            f"Checkpoint tokenizer/model is missing required ST tokens: {invalid_tokens}"
        )

    st_tower = model.get_model().st_tower


    st_config = st_tower.config
    st_config.st_patch_token = tokenizer.convert_tokens_to_ids([DEFAULT_ST_PATCH_TOKEN])[0]

    st_config.use_st_start_end = use_st_start_end
    if use_st_start_end:
        st_config.st_start_token, st_config.st_end_token = tokenizer.convert_tokens_to_ids(
            [DEFAULT_ST_START_TOKEN, DEFAULT_ST_END_TOKEN])

    res_data = []
    print(f'total: {len(prompt_file)}')
    with open(args.st_data_path, 'rb') as file:
        st_data_all = pickle.load(file)
    error_i = 0
    temp = 0
    output_file = osp.join(args.output_res_path, f'arxiv_test_res_{start_idx}_{end_idx}.json')

    for idx, instruct_item in tqdm(enumerate(prompt_file), total=len(prompt_file)):
        st_dict = load_st(idx, instruct_item, st_data_all)
        st_token_len = st_dict['st_token_len']
        st_data_x = st_dict['st_data_x']
        st_data_xd = st_dict['st_data_xd']
        st_data_xw = st_dict['st_data_xw']
        st_data_y = st_dict['st_data_y']
        region_start = st_dict['region_start']
        region_end = st_dict['region_end']
        std = st_dict['std']
        mean = st_dict['mean']

        st_data_xh_tmp = st_data_x * std + mean    
        st_data_xd_tmp = st_data_xd * std + mean   
        st_data_xw_tmp = st_data_xw * std + mean 
        
        st_data_xh_tmp = st_data_xh_tmp[0,:,region_start:region_end,:]
        st_data_xd_tmp = st_data_xd_tmp[0,:,region_start:region_end,:]
        st_data_xw_tmp = st_data_xw_tmp[0,:,region_start:region_end,:]

        qs = instruct_item["conversations"][0]["value"]
        original_sentence=instruct_item["conversations"][0]["value"]
        dataset = instruct_item["id"].split('_')[1]
        st_data_x_copy = copy.deepcopy(st_data_x).cuda()
        if dataset == "SD":
            graph = graphs[dataset]
            node_feature = graph["nodes_feature"].cuda()
            sp_matrix = graph["sp_matrix"].cuda()
            se_matrix = graph["se_matrix"].cuda()
            routing_info = select_routing_info(
                args, model.prompt_router_sd, model.model.st_tower,
                st_data_x_copy[..., :3], sp_matrix, se_matrix, 1,
                node_feature, region_start, region_end,
            )
            qs = replace_prompt_sd(
                routing_info, original_sentence,
                st_data_xh_tmp, st_data_xd_tmp, st_data_xw_tmp,
            )
        elif dataset == "SZ":
            graph = graphs[dataset]
            node_feature = graph["nodes_feature"].cuda()
            sp_matrix = graph["sp_matrix"].cuda()
            se_matrix = graph["se_matrix"].cuda()
            routing_info = select_routing_info(
                args, model.prompt_router_sz, model.model.st_tower,
                st_data_x_copy[..., :3], sp_matrix, se_matrix, 2,
                node_feature, region_start, region_end,
            )
            qs = replace_prompt_sz(
                routing_info, original_sentence,
                st_data_xh_tmp, st_data_xd_tmp, st_data_xw_tmp,
            )
        elif dataset in {"pems08", "pems03", "pems04"}:
            graph = graphs[dataset]
            node_feature = None
            sp_matrix = graph["sp_matrix"].cuda()
            se_matrix = graph["se_matrix"].cuda()
            routing_info = select_routing_info(
                args, model.prompt_router_pems08, model.model.st_tower,
                st_data_x_copy[..., :3], sp_matrix, se_matrix, 6,
                node_feature, region_start, region_end,
            )
            qs = replace_prompt_sd(
                routing_info, original_sentence,
                st_data_xh_tmp, st_data_xd_tmp, st_data_xw_tmp,
            )
        elif dataset == "urbanev":
            graph = graphs[dataset]
            node_feature = graph["nodes_feature"].cuda()
            sp_matrix = graph["sp_matrix"].cuda()
            se_matrix = graph["se_matrix"].cuda()
            routing_info = select_routing_info(
                args, model.prompt_router_urbanev, model.model.st_tower,
                st_data_x_copy[..., :3], sp_matrix, se_matrix, 5,
                node_feature, region_start, region_end,
            )
            qs = model.replace_prompt_urbanev(
                routing_info, original_sentence,
                st_data_xh_tmp, st_data_xd_tmp, st_data_xw_tmp,
            )
        else:
            raise ValueError(f"Unsupported evaluation dataset: {dataset}")

        patchlist = []
        cur_token_len = 12
        patchlist.append(cur_token_len)
        pre_token_len = 12
        replace_token = DEFAULT_ST_PATCH_TOKEN * cur_token_len
        replace_token = DEFAULT_ST_START_TOKEN + replace_token + DEFAULT_ST_END_TOKEN
        replace_token1 = DEFAULT_ST_PATCH_TOKEN * pre_token_len
        replace_token1 = DEFAULT_ST_START_TOKEN + replace_token1 + DEFAULT_ST_END_TOKEN
        qs = qs.replace(DEFAULT_STHIS_TOKEN, replace_token)
        qs = qs.replace(DEFAULT_STPRE_TOKEN, replace_token1)

        llama3_conversation = conv_templates["llama3_v2_1"]
        input_ids = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": llama3_conversation.system},
                {"role": "user", "content": qs},
            ],
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).cuda()
        attention_mask = (input_ids != tokenizer.pad_token_id).long()

        prediction_stopping_criteria = StoppingCriteriaList(
            [
                StopAfterSTPrediction(
                    st_config.st_start_token,
                    st_config.st_patch_token,
                    patch_count=12,
                )
            ]
        )


        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                st_data_x=st_data_x.cuda(),
                st_data_y=st_data_y.cuda(),
                region_start=region_start,
                region_end=region_end,
                do_sample=False,
                attention_mask = attention_mask,
                pad_token_id=tokenizer.pad_token_id,
                max_new_tokens=args.max_new_tokens,
                stopping_criteria=prediction_stopping_criteria,
                nodes_feature = node_feature,
                sp_matrix = sp_matrix,
                se_matrix = se_matrix,
                data_type = graph["st_encoder_type"],
                patchlist=patchlist,
                mean = mean.item(),
                std = std.item(),
                st_data_xd = st_data_xd,
                st_data_xw = st_data_xw)

            # Find the special tokens
            start_inx = torch.where(output_ids[0, :] == st_config.st_start_token)[0]
            end_inx = torch.where(output_ids[0, :] == st_config.st_end_token)[0]
            # Get hidden_states
            hidden_states = model.get_st_pre_res()
            hidden_states = torch.cat(hidden_states, dim=1)
            model.reset_st_pre_res()

            # Decode the token into the result
            batch_size = hidden_states.shape[0]

            feature_nums = 1
            input_token_len = input_ids.shape[1]
            generated_starts = start_inx[start_inx >= input_token_len]
            history_span_valid = (
                start_inx.numel() >= 1
                and end_inx.numel() >= 1
                and int(end_inx[0].item() - start_inx[0].item() - 1) == 12
            )
            generated_suffix_ids = output_ids[0, input_token_len:]
            generated_patch_offset = find_first_patch_run(
                generated_suffix_ids,
                st_config.st_patch_token,
                patch_count=12,
            )
            generated_patch_start = (
                input_token_len + generated_patch_offset
                if generated_patch_offset is not None
                else -1
            )
            generated_patch_end = generated_patch_start + 12
            generated_patch_tokens = output_ids[
                0, generated_patch_start:generated_patch_end
            ]
            prediction_span_valid = (
                generated_patch_offset is not None
                and generated_patch_tokens.numel() == 12
                and bool(
                    torch.all(
                        generated_patch_tokens == st_config.st_patch_token
                    ).item()
                )
            )
            if history_span_valid and prediction_span_valid:
                generated_ends = end_inx[end_inx > generated_patch_start]
                if (
                    generated_ends.numel() > 0
                    and int(generated_ends[0].item()) != generated_patch_end
                ):
                    print(
                        "warning: generated prediction span has an unexpected "
                        "closing ST token position; using the first 12 patches"
                    )
                st_pre_embs1 = hidden_states[:,
                            start_inx[0]+1:end_inx[0],
                            :].detach().reshape(batch_size, -1, feature_nums, model.config.hidden_size)
                head_mapping = {
                    "SD": (model.st_pred_linear_1, model.st_pred_linear_3, model.st_pred_linear_2),
                    "SZ": (model.st_pred_linear_4, model.st_pred_linear_6, model.st_pred_linear_5),
                    "pems08": (model.st_pred_linear_7, model.st_pred_linear_9, model.st_pred_linear_8),
                    "pems03": (model.st_pred_linear_7, model.st_pred_linear_9, model.st_pred_linear_8),
                    "pems04": (model.st_pred_linear_7, model.st_pred_linear_9, model.st_pred_linear_8),
                    "urbanev": (model.st_pred_linear_10, model.st_pred_linear_12, model.st_pred_linear_11),
                }
                history_head, future_head, output_head = head_mapping[dataset]
                st_pre_out1 = model.relu(history_head(st_pre_embs1))


                st_pre_embs2 = hidden_states[:,
                            generated_patch_start:generated_patch_end,
                            :].reshape(batch_size, -1, feature_nums, model.config.hidden_size)
                st_pre_out2 = model.relu(future_head(st_pre_embs2))

                st_pre_final = output_head(torch.cat([st_pre_out1, st_pre_out2], dim=-1)).reshape(st_pre_out1.shape[0],12,1,1)
                st_pre_infolow = st_pre_final[:, :, :, 0].squeeze().detach().cpu().tolist()


                x_in, y_in = st_data_x[:, :, region_start:region_end, 0].squeeze().tolist(), st_data_y[0, :, region_start:region_end,
                                                                                0].squeeze().tolist()
 
                input_token_len = input_ids.shape[1]
                n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
                if n_diff_input_output > 0:
                    print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
 
                res_data.append(
                    {
                        "id": instruct_item["id"],
                        "x_in": x_in,
                        "y_in": y_in,
                        "st_pre_infolow": st_pre_infolow,
                    }
                )
            else:
                generated_suffix = tokenizer.decode(
                    output_ids[0, input_ids.shape[1]:],
                    skip_special_tokens=False,
                )
                print(
                    "invalid_st_generation:",
                    {
                        "start_token_count": int(start_inx.numel()),
                        "end_token_count": int(end_inx.numel()),
                        "generated_token_count": int(
                            output_ids.shape[1] - input_ids.shape[1]
                        ),
                        "generated_suffix": generated_suffix,
                    },
                )
                error_i = error_i + 1
                print(error_i)

    with open(output_file, "w") as fout:
        json.dump(res_data, fout, indent=2)
    if not res_data:
        raise RuntimeError(
            "Evaluation produced no valid predictions; inspect ST token positions"
        )
    if torch.cuda.is_available():
        print(
            "evaluation_gpu_stats:",
            {
                "peak_allocated_gib": round(
                    torch.cuda.max_memory_allocated() / (1024 ** 3), 3
                ),
                "peak_reserved_gib": round(
                    torch.cuda.max_memory_reserved() / (1024 ** 3), 3
                ),
            },
        )
    return


if __name__ == "__main__":

    output_model='./checkpoints/your_model'
    datapath='./data/prompt_data/pems08_test.json'
    st_data_path='./data/prompt_data/pems08_test_pkl.pkl'
    res_path='./result_test/pems08'
    start_id=0
    end_id=None
    num_gpus=1

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default=output_model)
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Stage 1 checkpoint-N containing adapter and non-LoRA state",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="./checkpoints/llama3-8b",
        help="Base Llama model used to reconstruct --checkpoint",
    )
    parser.add_argument(
        "--fixed-prompt-index",
        type=int,
        choices=range(4),
        default=None,
        help=(
            "Use one fixed prompt variant for all four slots. This permits fair "
            "Stage 1 comparison of old checkpoints that did not save routers."
        ),
    )
    parser.add_argument("--prompting_file", type=str, default=datapath)
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--st_data_path", type=str, default=st_data_path)
    parser.add_argument("--output_res_path", type=str, default=res_path)
    parser.add_argument("--num_gpus", type=int, default=num_gpus)
    parser.add_argument("--max_new_tokens", type=int, default=256)

    parser.add_argument("--start_id", type=int, default=start_id)
    range_group = parser.add_mutually_exclusive_group()
    range_group.add_argument("--end_id", type=int, default=end_id)
    range_group.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Evaluate this many samples starting at --start_id",
    )

    args = parser.parse_args()

    total_samples = len(load_prompting_file(args.prompting_file))
    if args.num_samples is not None:
        if args.num_samples <= 0:
            parser.error("--num-samples must be positive")
        args.end_id = min(args.start_id + args.num_samples, total_samples)
    elif args.end_id is None:
        args.end_id = total_samples
    run_eval(args, args.num_gpus)
