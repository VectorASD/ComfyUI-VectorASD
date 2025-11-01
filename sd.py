# Copyright 2025 VectorASD
# Licensed under the Apache License, Version 2.0 (see LICENSE file in project root).
# This file may include modified components originally from ComfyUI (MIT License).

from pprint import pformat

from .umt5_xxl import patcher as umt5_xxl_patcher

import comfy.sd
from comfy.sd import CLIPType, TEModel
from comfy.cli_args import args
import folder_paths

import torch
import safetensors

DISABLE_MMAP = args.disable_mmap



def load_torch_file(ckpt, chunk, device=None, return_metadata=False):
    if device is None:
        device = torch.device("cpu")
    metadata = None
    assert ckpt.lower().endswith(".safetensors") or ckpt.lower().endswith(".sft")
    try:
        with safetensors.safe_open(ckpt, framework="pt", device=device.type) as f:
            sd = {}
            if chunk is None: keys = f.keys()
            else: keys = (key for key, _, _ in chunk)
            for key in keys:
                tensor = f.get_tensor(key)
                if DISABLE_MMAP:  # TODO: Not sure if this is the best way to bypass the mmap issues
                    tensor = tensor.to(device=device, copy=True)
                sd[key] = tensor
            if return_metadata:
                metadata = f.metadata()
    except Exception as e:
        if len(e.args) > 0:
            message = e.args[0]
            if "HeaderTooLarge" in message:
                raise ValueError("{}\n\nFile path: {}\n\nThe safetensors file is corrupt or invalid. Make sure this is actually a safetensors file and not a ckpt or pt or other filetype.".format(message, ckpt))
            if "MetadataIncompleteBuffer" in message:
                raise ValueError("{}\n\nFile path: {}\n\nThe safetensors file is corrupt/incomplete. Check the file size and make sure you have copied/downloaded it correctly.".format(message, ckpt))
        raise e
    return (sd, metadata) if return_metadata else sd



def load_clip(ckpt_paths, full_header, chunk, embedding_directory=None, clip_type=CLIPType.STABLE_DIFFUSION, model_options={}):
    clip_data = [load_torch_file(p, chunk=chunk) for p in ckpt_paths]
    return load_text_encoder_state_dicts(clip_data, full_header, embedding_directory=embedding_directory, clip_type=clip_type, model_options=model_options)



def load_text_encoder_state_dicts(clip_data=[], full_header={}, embedding_directory=None, clip_type=CLIPType.STABLE_DIFFUSION, model_options={}):
    class EmptyClass:
        pass

    for i in range(len(clip_data)):
        if "transformer.resblocks.0.ln_1.weight" in clip_data[i]:
            clip_data[i] = comfy.utils.clip_text_transformers_convert(clip_data[i], "", "")
        else:
            if "text_projection" in clip_data[i]:
                clip_data[i]["text_projection.weight"] = clip_data[i]["text_projection"].transpose(0, 1) # old models saved with the CLIPSave node

    tokenizer_data = {}
    clip_target = EmptyClass()
    clip_target.params = {}

    if len(clip_data) == 1:
        te_model = comfy.sd.detect_te_model(full_header)
        if te_model == TEModel.T5_XXL:
            if clip_type == CLIPType.WAN:
                clip_target.clip = comfy.text_encoders.wan.te(**comfy.sd.t5xxl_detect((full_header,))) # {'dtype_t5': torch.float16, 't5xxl_scaled_fp8': torch.float8_e4m3fn}
                if "spiece_model" in clip_data[0]:
                    clip_target.tokenizer = comfy.text_encoders.wan.WanT5Tokenizer
                    tokenizer_data["spiece_model"] = clip_data[0]["spiece_model"]
                else: clip_target.tokenizer = lambda embedding_directory, tokenizer_data: None

                umt5_xxl_patcher(clip_data[0], model_options)
            else:
                raise ValueError("Support only clip_type == CLIPType.WAN")
        else:
            raise ValueError("Support only te_model == TEModel.T5_XXL")
    else:
        raise ValueError("Support only len(ckpt_paths) == 1")

    parameters = 0
    for c in clip_data:
        parameters += comfy.utils.calculate_parameters(c)
        tokenizer_data, model_options = comfy.text_encoders.long_clipl.model_options_long_clip(c, tokenizer_data, model_options)

    #print(tokenizer_data)
    #print(vars(clip_target))
    #print(parameters)
    #print(model_options)
    #exit()
    import logging

    clip = comfy.sd.CLIP(clip_target, embedding_directory=embedding_directory, parameters=parameters, tokenizer_data=tokenizer_data, model_options=model_options)
    for c in clip_data:
        # c = {}
        m, u = clip.load_sd(c)
        m = model_options["warn_filter"](m)
        if m: logging.warning("clip missing: {}".format(pformat(m)))
        if u: logging.debug("clip unexpected: {}".format(pformat(u)))

    # print(clip.cond_stage_model)

    return clip



def load_clip_direct(ckpt_paths, full_header, chunk, type="stable_diffusion", device="default"):
    clip_type_enum = getattr(CLIPType, type.upper(), CLIPType.STABLE_DIFFUSION)
    model_options = {}
    if device == "cpu":
        model_options["load_device"] = model_options["offload_device"] = torch.device("cpu")
    clip = load_clip(
        ckpt_paths=ckpt_paths,
        full_header=full_header,
        chunk=chunk,
        embedding_directory=folder_paths.get_folder_paths("embeddings"),
        clip_type=clip_type_enum,
        model_options=model_options
    )
    return clip, model_options["parts"]
