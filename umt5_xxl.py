# Copyright 2025 VectorASD
# Licensed under the Apache License, Version 2.0 (see LICENSE file in project root).
# This file may include modified components originally from ComfyUI (MIT License).

from comfy import model_management
from comfy.ldm.modules.attention import optimized_attention_for_device

import torch



def patcher(clip_data, model_options):
    block_nums = set()
    for label in clip_data:
        if label.startswith("encoder.block."):
            num = int(label.split(".")[2])
            block_nums.add(num)

    blocks = _min = 0
    if block_nums:
        _min, _max, blocks = min(block_nums), max(block_nums), len(block_nums)
        if _max - _min + 1 != blocks: raise ValueError("Holes between the blocks: {block_nums}")

    if _min:
        shifted_clips = {}
        for label, tensor in clip_data.items():
            if label.startswith("encoder.block."):
                split = label.split(".")
                split[2] = str(int(split[2]) - _min)
                # print(label, "->", ".".join(split))
                label = ".".join(split)
            shifted_clips[label] = tensor
        clip_data.clear()
        clip_data.update(shifted_clips)
        # print(clip_data.keys())

    is_spiece_model = "spiece_model" in clip_data
    is_shared       = "shared.weight" in clip_data
    is_final        = "encoder.final_layer_norm.weight" in clip_data

    parts = set()
    if is_spiece_model: parts.add("spiece_model")
    if is_shared:       parts.add("shared")
    if blocks:          parts.add("encoder")
    if is_final:        parts.add("final_layer_norm")
    model_options["parts"] = parts

    model_options["umt5xxl_model_config"] = {
        "num_layers": blocks,
        "vocab_size": 256384 if is_shared else 0,
    }

    ignore_list = []
    if not is_final: ignore_list.append("encoder.final_layer_norm.weight")
    if not is_shared: ignore_list.append("shared.weight")
    ignore_list = tuple(ignore_list)

    model_options["warn_filter"] = lambda missings: tuple(missing for missing in missings if missing not in ignore_list)



def encode_from_tokens(clip, data, return_pooled=False, return_dict=False, parts=set()):
    # comfy.sd: class CLIP:
    clip.cond_stage_model.reset_clip_options()

    if clip.layer_idx is not None:
        clip.cond_stage_model.set_clip_options({"layer": clip.layer_idx})

    if return_pooled == "unprojected":
        clip.cond_stage_model.set_clip_options({"projected_pooled": False})

    clip.load_model()

    # o = self.cond_stage_model.encode_token_weights(tokens)
    # comfy.sd1_clip: class SD1ClipModel(torch.nn.Module): def encode_token_weights(self, token_weight_pairs)

    clip_model = clip.cond_stage_model
    model = getattr(clip_model, clip_model.clip)

    t5 = model.transformer
    t5stack = t5.encoder

    if "spiece_model" in parts:
        assert data[0] == "TEXT", f"required type 'TEXT', but arrived {data[0]}"
        TYPE, text = data

        tokenizer = clip.tokenizer # comfy.text_encoders.wan.WanT5Tokenizer
        # print("clip:",      tokenizer.clip)      # umt5xxl
        # print("clip_name:", tokenizer.clip_name) # clip_umt5xxl
        tokenizer = getattr(tokenizer, tokenizer.clip) # comfy.text_encoders.wan.UMT5XXlTokenizer
        tokenizer = tokenizer.tokenizer # comfy.text_encoders.spiece_tokenizer.SPieceTokenizer
        tokenizer = tokenizer.tokenizer # sentencepiece.SentencePieceProcessor; proxy of <Swig Object of type 'sentencepiece::SentencePieceProcessor *'

        tokens = clip.tokenize(text)

        for _model, batch in tokens.items():
            for tokenz in batch:
                size = len(tokenz)
                tokenz    = tuple((token, w) for token, w in tokenz if token) # token = 0 = <pad>
                input_ids = tuple((tokenizer.id_to_piece(token), w) for token, w in tokenz)
                print(f"TOKENS({_model})(size: {size}):", tokenz)
                print("  input_ids:", input_ids)

        data = "TOKENS", tokens

    if "shared" in parts:
        assert data[0] == "TOKENS", f"required type 'TOKENS', but arrived {data[0]}"
        TYPE, tokens = data

        token_weight_pairs = tokens[clip_model.clip_name]

        # o = model.encode_token_weights(token_weight_pairs)
        # comfy.sd1_clip: class ClipTokenWeightEncoder: def encode_token_weights(self, token_weight_pairs)

        to_encode = list()
        max_token_len = 0
        has_weights = False
        for x in token_weight_pairs:
            tokens = list(map(lambda a: a[0], x))
            max_token_len = max(len(tokens), max_token_len)
            has_weights = has_weights or not all(map(lambda a: a[1] == 1.0, x))
            to_encode.append(tokens)

        sections = len(to_encode)
        if has_weights or sections == 0:
            if hasattr(model, "gen_empty_tokens"):
                to_encode.append(model.gen_empty_tokens(model.special_tokens, max_token_len))
            else:
                to_encode.append(gen_empty_tokens(model.special_tokens, max_token_len))

        # o = model.encode(to_encode)
        # comfy.sd1_clip: class SDClipModel(torch.nn.Module, ClipTokenWeightEncoder): def encode(self, tokens)

        # o = model(to_encode)
        # comfy.sd1_clip: class SDClipModel(torch.nn.Module, ClipTokenWeightEncoder): def forward(self, tokens)

        device = model.transformer.get_input_embeddings().weight.device # cuda:0
        embeds, attention_mask, num_tokens, embeds_info = model.process_tokens(to_encode, device)

        data = "EMBEDS", embeds, None, attention_mask, (sections, has_weights) # num_tokens and embeds_info unused

    if "encoder" in parts:
        assert data[0] == "EMBEDS", f"required type 'EMBEDS', but arrived {data[0]}"
        TYPE, embeds, _, attention_mask, misc = data

        attention_mask_model = attention_mask if model.enable_attention_masks else None
        intermediate_output = "all" if model.layer == "all" else model.layer_idx

        # outputs = model.transformer(None, attention_mask_model, embeds=embeds, num_tokens=num_tokens, intermediate_output=intermediate_output, final_layer_norm_intermediate=model.layer_norm_hidden_state, dtype=torch.float32, embeds_info=embeds_info)
        # comfy.text_encoders.t5: class T5(torch.nn.Module): def forward(self, input_ids, attention_mask, embeds=None, num_tokens=None, **kwargs)

        # if input_ids is None:
        x = embeds
        # else:
        #     x = t5.shared(input_ids, out_dtype=kwargs.get("dtype", torch.float32))
        if t5.dtype not in (torch.float32, torch.float16, torch.bfloat16):
            x = torch.nan_to_num(x) #Fix for fp8 T5 base

        # outputs = t5.encoder(x, attention_mask=attention_mask_model, **kwargs)
        # comfy.text_encoders.t5: class T5Stack(torch.nn.Module): def forward(self, x, attention_mask=None, intermediate_output=None, final_layer_norm_intermediate=True, dtype=None, embeds_info=[])

        # kwargs (повторяюшиеся переменные: intermediate_output и embeds_info):
        final_layer_norm_intermediate = model.layer_norm_hidden_state
        dtype = torch.float32

        mask = None
        if attention_mask_model is not None:
            mask = 1.0 - attention_mask_model.to(x.dtype).reshape((attention_mask_model.shape[0], 1, -1, attention_mask_model.shape[-1])).expand(attention_mask_model.shape[0], 1, attention_mask_model.shape[-1], attention_mask_model.shape[-1])
            mask = mask.masked_fill(mask.to(torch.bool), -torch.finfo(x.dtype).max)

        intermediate = None
        optimized_attention = optimized_attention_for_device(x.device, mask=attention_mask_model is not None, small_input=True)
        past_bias = None

        if intermediate_output is not None:
            if intermediate_output < 0:
                intermediate_output = len(t5stack.block) + intermediate_output

        for i, l in enumerate(t5stack.block):
            x, past_bias = l(x, mask, past_bias, optimized_attention)
            if i == intermediate_output:
                intermediate = x.clone()

        data = "EMBEDS", x, intermediate, attention_mask, misc

    if "final_layer_norm" in parts:
        assert data[0] == "EMBEDS", f"required type 'EMBEDS', but arrived {data[0]}"
        TYPE, embeds, intermediate, attention_mask, (sections, has_weights) = data

        x = t5stack.final_layer_norm(embeds)
        if intermediate is not None and final_layer_norm_intermediate:
            intermediate = t5stack.final_layer_norm(intermediate)
        outputs = x, intermediate

        # end
        # end

        if model.layer == "last":
            z = outputs[0].float()
        else:
            z = outputs[1].float()

        if model.zero_out_masked:
            z *= attention_mask.unsqueeze(-1).float()

        pooled_output = None
        if len(outputs) >= 3:
            if not model.return_projected_pooled and len(outputs) >= 4 and outputs[3] is not None:
                pooled_output = outputs[3].float()
            elif outputs[2] is not None:
                pooled_output = outputs[2].float()

        extra = {}
        if model.return_attention_masks:
            extra["attention_mask"] = attention_mask

        if len(extra) > 0: o = z, pooled_output, extra
        else:              o = z, pooled_output

        # end

        out, pooled = o[:2]

        if pooled is not None:
            first_pooled = pooled[0:1].to(model_management.intermediate_device())
        else:
            first_pooled = pooled

        output = []
        for k in range(0, sections):
            z = out[k:k+1]
            if has_weights:
                z_empty = out[-1]
                for i in range(len(z)):
                    for j in range(len(z[i])):
                        weight = token_weight_pairs[k][j][1]
                        if weight != 1.0:
                            z[i][j] = (z[i][j] - z_empty[j]) * weight + z_empty[j]
            output.append(z)

        if (len(output) == 0):
            r = (out[-1:].to(model_management.intermediate_device()), first_pooled)
        else:
            r = (torch.cat(output, dim=-2).to(model_management.intermediate_device()), first_pooled)

        if len(o) > 2:
            extra = {}
            for k in o[2]:
                v = o[2][k]
                if k == "attention_mask":
                    v = v[:sections].flatten().unsqueeze(dim=0).to(model_management.intermediate_device())
                extra[k] = v

            r = r + (extra,)
        o = r

        # end
        # end

        cond, pooled = o[:2]
        if return_dict:
            out = {"cond": cond, "pooled_output": pooled}
            if len(o) > 2:
                for k in o[2]:
                    out[k] = o[2][k]
            clip.add_hooks_to_dict(out)

            data = "FINAL", out
        elif return_pooled:
            data = "FINAL", cond, pooled
        else:
            data = "FINAL", cond

    return data



def encode_from_tokens_scheduled(clip, data, unprojected=False, add_dict: dict[str]={}, show_pbar=True, parts=set()):
    all_cond_pooled: list[tuple[torch.Tensor, dict[str]]] = []
    all_hooks = clip.patcher.forced_hooks
    if all_hooks is None or not clip.use_clip_schedule:
        # if no hooks or shouldn't use clip schedule, do unscheduled encode_from_tokens and perform add_dict
        return_pooled = "unprojected" if unprojected else True
        data = encode_from_tokens(clip, data, return_pooled=return_pooled, return_dict=True, parts=parts)

        if data[0] != "FINAL":
            return data
        TYPE, pooled_dict = data

        cond = pooled_dict.pop("cond")
        # add/update any keys with the provided add_dict
        pooled_dict.update(add_dict)
        all_cond_pooled.append([cond, pooled_dict])
    else:
        raise NotImplementedError("clip_schedule is not supported")

    return "SCHEDULED", all_cond_pooled



""" umt5_xxl_fp8_e4m3fn_scaled.safetensors

blocks = 1
|        *=•o                :===°                 |
|    ·•=•·+*+             *==o                     |
| o=•oo*==:           -===.                        |
| •===·           :===°                       °==• |
|              o=•+                         •▓▒*=* |
|           ·░░:                        ·▒@█•:     |
|         +O•                       o=*@%+         |
|      :░O-                     o•==:▒░            |
|    •░*                     *•=: :░░              |
| +░•.                    •••.  +O=                |
| +                   .░░•    *░o               ░░ |
|                  ·•░=    ·░O:              =••   |
|               -░••      °=              -░░·     |
|            *=•+                       =░+        |
|         •░•·                       :O░·          |
|      -OO·                        =░*             |
|    *▒=                        ·OO:               |
| .▒▒-                        o░=                  |
| ░                         ░░+                 =░ |
|                        °░░                 :OO:  |
|                      =O+                 *O=     |
|                  :===:                ·●░-       |
|               o==+                 ===o          |
|           ·=••·                -===              |
pooled_output: None
tensor (cpu) (F32[1, 512, 4096]) sha256: c988b24b02e73c39...



blocks = 24
|        •░*          :******:                     |
|     -░O.     :******°                            |
| ****° .******+                               .** |
| ******o                                o*****o . |
|                                 ·******.    -░•* |
|                           o*****+         ░░o    |
|                      :░***.            =░•       |
|                    =░*              o░•·         |
|                  ░▒: ************+░░°            |
| :=************•█@=***          =░=               |
| *           *░=             o░░.             ·•• |
|           O░-            .░░-              •░*   |
|        -OO.             °•              =••.     |
|      *░=                             =•░.        |
|    ░▒-                            =••.           |
| -░░.                           o░•.              |
| o                           =••:              =O |
|                          o•░.              -░O:  |
|                       ·O░-               ░Oo     |
|                     *O=               °░░        |
|                  :░O-               •O+          |
|                =░=               ·▒O.            |
|             o░•·               ░░*               |
|          :•O-              °=░░                  |
pooled_output: None
tensor (cpu) (F32[1, 512, 4096]) sha256: 3ca62203c0864acc...

"""
