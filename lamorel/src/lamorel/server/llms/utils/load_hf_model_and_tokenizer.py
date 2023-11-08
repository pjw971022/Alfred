from enum import Enum

# from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer, AutoConfig

import os
import math
import torch
from types import MethodType
from typing import TYPE_CHECKING, Literal, Optional, Tuple

from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PretrainedConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase
)
from transformers.models.llama import modeling_llama as LlamaModule
from transformers.utils import check_min_version
from transformers.utils.versions import require_version

try:
    from transformers.integrations import is_deepspeed_zero3_enabled
except ImportError: # https://github.com/huggingface/transformers/releases/tag/v4.33.1
    from transformers.deepspeed import is_deepspeed_zero3_enabled

# import ipdb;ipdb.set_trace()
from ..llmtuner.extras.logging import reset_logging, get_logger
from ..llmtuner.extras.misc import count_parameters
from ..llmtuner.extras.patches import llama_patch as LlamaPatches
from ..llmtuner.extras.save_and_load import load_valuehead_params
from ..llmtuner.hparams import FinetuningArguments
# from ..llmtuner.tuner.core.adapter import init_adapter
# from ..llmtuner.tuner.core.utils import prepare_model_for_training

class ModelTypesEnum(Enum):
    causal = AutoModelForCausalLM
    seq2seq = AutoModelForSeq2SeqLM


def load_hf_model_and_tokenizer_original(type, path, pretrained):
    print("Loading model {}".format(path))
    tokenizer = AutoTokenizer.from_pretrained(path)

    # Select class according to type
    config = AutoConfig.from_pretrained(path)

    n_layers_key = 'num_hidden_layers'
    if hasattr(config, "attribute_map") and n_layers_key in config.attribute_map:
        n_layers_key = config.attribute_map[n_layers_key]

    n_layers = getattr(config, n_layers_key)
    model_class = ModelTypesEnum[type].value
    if pretrained:
        model_method = lambda **kwargs: model_class.from_pretrained(path, **kwargs)

    else:
        model_method = lambda **kwargs: model_class.from_config(config, **kwargs)

    return tokenizer, model_method, n_layers




def load_hf_model_and_tokenizer_complex(
    model_args: "ModelArguments",
    # finetuning_args: "FinetuningArguments",
    is_trainable: Optional[bool] = False,
    # stage: Optional[Literal["pt", "sft", "rm", "ppo"]] = "sft"
) -> Tuple[PreTrainedModel, "PreTrainedTokenizer"]:
    r"""
    Loads pretrained model and tokenizer.

    Support both training and inference.
    """
    # if (not is_trainable) and model_args.checkpoint_dir is None:
    #     # logger.warning("Checkpoint is not found at evaluation, load the original model.")
    #     finetuning_args = FinetuningArguments(finetuning_type="none")
    config_kwargs = {
        "trust_remote_code": True,
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        use_fast=model_args.use_fast_tokenizer,
        padding_side="right", # training with left-padded tensors in fp16 precision may cause overflow
        **config_kwargs
    )

    # if finetuning_args.finetuning_type != "lora" and model_args.checkpoint_dir is not None:
    #     model_to_load = model_args.checkpoint_dir[0]
    # else:
    model_to_load = model_args.model_name_or_path

    config = AutoConfig.from_pretrained(model_to_load, **config_kwargs)

    # Fix tokenizer (for ChatGLM2)
    # if getattr(config, "model_type", None) == "chatglm":
    #     tokenizer._pad = MethodType(PreTrainedTokenizerBase._pad, tokenizer)

    # # Fix config (for Qwen)
    # if getattr(config, "model_type", None) == "qwen":
    #     setattr(config, "fp16", model_args.compute_dtype == torch.float16)
    #     setattr(config, "bf16", model_args.compute_dtype == torch.bfloat16)
    #     setattr(config, "fp32", model_args.compute_dtype == torch.float32)

    # Set RoPE scaling
    if model_args.rope_scaling is not None:
        if hasattr(config, "use_dynamic_ntk"): # for Qwen models
            if is_trainable:
                pass
                # logger.warning("Qwen model does not support RoPE scaling in training.")
            else:
                setattr(config, "use_dynamic_ntk", True)
                setattr(config, "use_logn_attn", True)
                # logger.info("Using dynamic NTK scaling.")

        elif hasattr(config, "rope_scaling"): # for LLaMA and Falcon models
            require_version("transformers>=4.31.0", "RoPE scaling requires transformers>=4.31.0")
            if is_trainable:
                # if model_args.rope_scaling == "dynamic":
                    # logger.warning(
                    #     "Dynamic NTK may not work well with fine-tuning. "
                    #     "See: https://github.com/huggingface/transformers/pull/24653"
                    # )

                current_max_length = getattr(config, "max_position_embeddings", None)
                if current_max_length and model_args.model_max_length > current_max_length:
                    scaling_factor = float(math.ceil(model_args.model_max_length / current_max_length))
                else:
                    # logger.warning("Input length is smaller than max length. Consider increase input length.")
                    scaling_factor = 1.0
            else:
                scaling_factor = 2.0

            setattr(config, "rope_scaling", {"type": model_args.rope_scaling, "factor": scaling_factor})
            # logger.info("Using {} scaling strategy and setting scaling factor to {}".format(
            #     model_args.rope_scaling, scaling_factor
            # ))

        else:
            pass
            # logger.warning("Current model does not support RoPE scaling.")

    # Set FlashAttention-2
    if model_args.flash_attn:
        if getattr(config, "model_type", None) == "llama":
            LlamaModule.LlamaAttention = LlamaPatches.LlamaFlashAttention2
            LlamaModule.LlamaModel._prepare_decoder_attention_mask = (
                LlamaPatches._prepare_decoder_attention_mask
            )
            # logger.info("Using FlashAttention-2 for faster training and inference.")
        # else:
        #     logger.warning("Current model does not support FlashAttention-2.")
    elif is_trainable and model_args.shift_attn and getattr(config, "model_type", None) == "llama":
        LlamaModule.LlamaAttention = LlamaPatches.LlamaShiftShortAttention
        # logger.warning("Using `--flash_attn` for faster training in large context length.")

    # Quantization configurations (using bitsandbytes library).
    is_mergeable = True
    if model_args.quantization_bit is not None:
        if is_deepspeed_zero3_enabled():
            raise ValueError("DeepSpeed ZeRO-3 is incompatible with quantization.")

        if model_args.quantization_bit == 8:
            require_version("bitsandbytes>=0.37.0", "To fix: pip install bitsandbytes>=0.37.0")
            config_kwargs["load_in_8bit"] = True
            config_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)

        elif model_args.quantization_bit == 4:
            require_version("bitsandbytes>=0.39.0", "To fix: pip install bitsandbytes>=0.39.0")
            config_kwargs["load_in_4bit"] = True
            config_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=model_args.compute_dtype,
                bnb_4bit_use_double_quant=model_args.double_quantization,
                bnb_4bit_quant_type=model_args.quantization_type
            )

        is_mergeable = False
        config_kwargs["device_map"] = {"": int(os.environ.get("LOCAL_RANK", "0"))} if is_trainable else "auto"
        # logger.info("Quantizing model to {} bit.".format(model_args.quantization_bit))

    # Load and prepare pre-trained models (without valuehead).
    model = AutoModelForCausalLM.from_pretrained(
        model_to_load,
        config=config,
        torch_dtype=model_args.compute_dtype,
        low_cpu_mem_usage=(not is_deepspeed_zero3_enabled()),
        **config_kwargs
    )

    # Set shift short attention (S^2-Attn)
    if is_trainable and model_args.shift_attn:
        if getattr(config, "model_type", None) == "llama":
            setattr(model, "shift_ratio", 0.25)
            # logger.info("Using shift short attention proposed by LongLoRA.")
        else:
            pass
            # logger.warning("Current model does not support shift short attention.")

    # Disable custom generate method (for Qwen and Baichuan2)
    if isinstance(model, PreTrainedModel) and "GenerationMixin" not in str(model.generate.__func__):
        model.generate = MethodType(PreTrainedModel.generate, model)

    # Fix LM head (for ChatGLM2)
    if getattr(config, "model_type", None) == "chatglm":
        setattr(model, "lm_head", model.transformer.output_layer)

    # Register auto class to save the custom code files.
    if isinstance(config, PretrainedConfig) and "AutoConfig" in getattr(config, "auto_map", {}):
        config.__class__.register_for_auto_class()
    if isinstance(model, PreTrainedModel) and "AutoModelForCausalLM" in getattr(config, "auto_map", {}):
        model.__class__.register_for_auto_class()
    if isinstance(tokenizer, PreTrainedTokenizerBase) and "AutoTokenizer" in tokenizer.init_kwargs.get("auto_map", {}):
        tokenizer.__class__.register_for_auto_class()

    return model, tokenizer
