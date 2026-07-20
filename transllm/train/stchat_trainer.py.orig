import os
import torch
import torch.nn as nn
import sys
from transformers import Trainer
from typing import Dict, Optional, Sequence


def unwrap_model(model: nn.Module) -> nn.Module:
    """
    Recursively unwraps a model from potential containers (as used in distributed training).

    Args:
        model (`torch.nn.Module`): The model to unwrap.
    """
    # since there could be multiple levels of wrapping, unwrap recursively
    if hasattr(model, "module"):
        return unwrap_model(model.module)
    else:
        return model

from torch.utils.data import DataLoader
class STChatTrainer(Trainer):
    loss_component_names = (
        "language_loss",
        "regression_loss",
        "normalized_regression_loss",
    )

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        loss, outputs = super().compute_loss(
            model,
            inputs,
            return_outputs=True,
            **kwargs,
        )

        sources = inputs.get("sources")
        dataset = None
        if sources:
            try:
                dataset = sources[0][0]["id"].split("_")[1]
            except (IndexError, KeyError, TypeError):
                dataset = None

        if dataset is not None and hasattr(outputs, "get"):
            totals = getattr(self, "_loss_component_totals", None)
            counts = getattr(self, "_loss_component_counts", None)
            if totals is None or counts is None:
                totals = self._loss_component_totals = {}
                counts = self._loss_component_counts = {}
            for name in self.loss_component_names:
                value = outputs.get(name)
                if value is None:
                    continue
                key = (dataset, name)
                totals[key] = totals.get(key, 0.0) + float(
                    value.detach().float().item()
                )
                counts[key] = counts.get(key, 0) + 1

        return (loss, outputs) if return_outputs else loss

    def log(self, logs, *args, **kwargs):
        # Add averages to the Trainer's normal logging event rather than
        # emitting one record for every gradient-accumulation micro-batch.
        totals = getattr(self, "_loss_component_totals", {})
        counts = getattr(self, "_loss_component_counts", {})
        if "loss" in logs and totals:
            logs = dict(logs)
            for (dataset, name), total in sorted(totals.items()):
                count = counts[(dataset, name)]
                logs[f"{name}/{dataset}"] = round(total / count, 6)
            totals.clear()
            counts.clear()
        return super().log(logs, *args, **kwargs)

    def get_train_dataloader(self):
        dataloader_kwargs = dict(
            dataset=self.train_dataset,
            batch_size=self.args.train_batch_size,
            shuffle=False,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )
        if self.args.dataloader_num_workers > 0:
            dataloader_kwargs["persistent_workers"] = (
                self.args.dataloader_persistent_workers
            )
            if self.args.dataloader_prefetch_factor is not None:
                dataloader_kwargs["prefetch_factor"] = (
                    self.args.dataloader_prefetch_factor
                )
        return DataLoader(**dataloader_kwargs)
    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        if getattr(self.args, 'tune_st_mlp_adapter', False):
            _state_dict = state_dict
            if _state_dict is None:
                # Only save the model itself if we are using distributed training
                model_to_save = unwrap_model(self.model)
                _state_dict = model_to_save.state_dict()

            weight_to_save = {}
            keys_to_match = ['st_projector']
            for k, v in _state_dict.items():
                if any(key_match in k for key_match in keys_to_match):
                    weight_to_save[k] = v

            print('output_dir', output_dir)
            current_folder = output_dir.split('/')[-1]
            if "\\" in current_folder:
                current_folder =  current_folder.split('\\')[-1]
            parent_folder = os.path.dirname(output_dir)
            if current_folder.startswith('checkpoint-'):
                mm_projector_folder = os.path.join(parent_folder, "st_projector")
                os.makedirs(mm_projector_folder, exist_ok=True)
                print('up', mm_projector_folder, current_folder)
                torch.save(weight_to_save, os.path.join(mm_projector_folder, f'{current_folder}.bin'))
            else:
                os.makedirs(output_dir, exist_ok=True)
                print('down', current_folder)
                torch.save(weight_to_save, os.path.join(output_dir, f'st_projector.bin'))
            # sys.stdout = open(os.devnull, 'w')

        super(STChatTrainer, self)._save(output_dir, state_dict)

        # PEFT checkpoints contain adapters but omit the other trainable modules
        # (prediction heads, projector, lm_head, or routers). Save them beside
        # every Trainer checkpoint so resume is mathematically complete.
        output_dir = output_dir or self.args.output_dir
        model_to_save = unwrap_model(self.model)
        # Frozen routers still affect prompt selection in Stage 1 and therefore
        # belong to the mathematical checkpoint even though they have no grad.
        inference_router_names = (
            "prompt_router_sd.",
            "prompt_router_pems08.",
            "prompt_router_sz.",
            "prompt_router_urbanev.",
        )
        trainable_names = {
            name
            for name, parameter in model_to_save.named_parameters()
            if (
                (parameter.requires_grad and "lora_" not in name)
                or any(router_name in name for router_name in inference_router_names)
            )
        }
        complete_state = state_dict if state_dict is not None else model_to_save.state_dict()
        non_lora_state = {
            name: complete_state[name].detach().cpu()
            for name in trainable_names
            if name in complete_state
        }
        missing = sorted(trainable_names.difference(non_lora_state))
        if missing:
            raise RuntimeError(
                f"Could not save all trainable non-LoRA parameters: {missing}"
            )
        os.makedirs(output_dir, exist_ok=True)
        torch.save(
            non_lora_state,
            os.path.join(output_dir, "non_lora_trainables.bin"),
        )

