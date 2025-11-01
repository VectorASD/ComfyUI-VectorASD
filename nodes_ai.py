# Copyright 2025 VectorASD
# Licensed under the Apache License, Version 2.0

import struct
import json
from pprint import pprint
import os
import pickle

from .tensor_tree_builder import header_to_tree
from . import utils
from . import umt5_xxl
from . import sd

import folder_paths
from comfy.utils import ProgressBar
from comfy.comfy_types import IO, ComfyNodeABC, InputTypeDict

import torch



class CLIPLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "clip_name": (folder_paths.get_filename_list("text_encoders"), ),
                "type": (["stable_diffusion", "stable_cascade", "sd3", "stable_audio", "mochi", "ltxv", "pixart", "cosmos", "lumina2", "wan", "hidream", "chroma", "ace", "omnigen2", "qwen_image", "hunyuan_image"], ),
                "chunk_size_gb": ("FLOAT", {"default": 5.0, "min": 1.0, "max": 20.0, "step": 0.5, "tooltip": "Максимальный размер одной части модели в гигабайтах"})
            },
            "optional": {
                "device": (["default", "cpu"], {"advanced": True}),
            }
        }

    RETURN_TYPES = ("CLIP_META",)
    RETURN_NAMES = ("meta clip",)
    FUNCTION = "load_clip_meta"
    CATEGORY = "VectorASD 🔥/AI"

    def load_clip_meta(self, clip_name, type="stable_diffusion", chunk_size_gb=5.0, device="default"):
        clip_path = folder_paths.get_full_path_or_raise("text_encoders", clip_name)
        chunk_specs, header_len = self.split_model(clip_path, chunk_size_gb)

        return ({
            "clip_path": clip_path,
            "clip_chunks": chunk_specs,
            "clip_type": type,
            "device": device,
            "clip_header_offset": header_len + 8,
        },)

    def split_model(self, clip_path, chunk_size_gb):
        chunk_limit_bytes = int(chunk_size_gb * 1024**3)

        header, header_len = utils.load_header(clip_path)

        root = header_to_tree(header)
        required    = [] # "scaled_fp" (учавствует и в first_items, и в parts, и в last_items)
        first_items = [] # "spiece_model", "shared."
        parts       = [] # "encoder.block."
        last_items  = [] # "encoder.final_layer_norm."
        for name, node in root.children.items():
            if name == "scaled_fp":
                node.collect(required.append)
            elif name in ("shared.", "spiece_model"):
                node.collect(first_items.append)
            elif name == "encoder.":
                for block in node.children["encoder.block."].children.values():
                    tensors = []
                    block.collect(tensors.append)
                    parts.append(tensors)
                node.children["encoder.final_layer_norm."].collect(last_items.append)

        if False: # debug
            for item in required: print("required:", item[0])
            for item in first_items: print("first_item:", item[0])
            for part in parts:
                print("~~~")
                for item in part: print(item[0])
            # print("~~~") разрез не имеет смысла, т.к. encoder.final_layer_norm.weight занимает всего 8 Kb
            for item in last_items: print("last_item:", item[0])
            exit()

        def load(items):
            nonlocal current_size
            for label, value, n_params in items:
                size = utils.name_to_size[value.get("dtype", '?')] * n_params
                current_size += size
                current_chunk.append((label, value, n_params))

        chunk_specs = []
        current_chunk = []
        current_size = 0

        load(required)
        load(first_items)

        for part in parts:
            batch = []
            for label, value, n_params in part:
                size = utils.name_to_size[value.get("dtype", '?')] * n_params
                # print(label, "|", size)
                current_size += size
                batch.append((label, value, n_params))

            if current_size > chunk_limit_bytes and current_chunk:
                chunk_specs.append(current_chunk)
                current_chunk = []
                current_size = size
                load(required)

            current_chunk.extend(batch)

        load(last_items)
        if current_chunk:
            chunk_specs.append(current_chunk)

        if False: # debug
            for spec in chunk_specs:
                print("~~~")
                for item in spec: print(item[0])
            exit()

        return chunk_specs, header_len



class CLIPSplitter:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "meta clip": ("CLIP_META", {
                    "tooltip": "CLIP_META containing the original safetensors path and chunked tensor layout."
                })
            }
        }

    RETURN_TYPES = ("CLIP_SPLIT",)
    RETURN_NAMES = ("clip split",)
    FUNCTION = "save_chunks"
    CATEGORY = "VectorASD 🔥/AI"
    DESCRIPTION = """
Splits a safetensors file into multiple chunked files
with recalculated headers and minimal memory usage.
DEPRECATED! You no longer need to cut the model into pieces
to load it. Use the Meta CLIP Text Encode immediately ;'-}
    """

    def save_chunks(self, **kwargs):
        meta_clip = kwargs["meta clip"]
        clip_path     = meta_clip["clip_path"]
        clip_chunks   = meta_clip["clip_chunks"]
        header_offset = meta_clip["clip_header_offset"]
        clip_type     = meta_clip.get("clip_type", "stable_diffusion")
        device        = meta_clip.get("device",    "default")

        base_name    = os.path.basename(clip_path)
        name_root, _ = os.path.splitext(base_name)
        split_dir    = os.path.join("models", "split")
        os.makedirs(split_dir, exist_ok=True)

        split_paths = []

        pbar = ProgressBar(len(clip_chunks) + sum(len(chunk) for chunk in clip_chunks)) # заголовков + блоков весов

        with open(clip_path, "rb") as f_in:
            for i, chunk in enumerate(clip_chunks):
                split_filename = f"{name_root}.{i}.safetensors"

                meta_path = os.path.join(split_dir, f"{name_root}.{i}.meta")
                split_path = os.path.join(split_dir, split_filename)
                meta = {"filename": split_filename, "chunk": chunk}

                if os.path.exists(meta_path):
                    with open(meta_path, "rb") as f_meta:
                        prev_meta = pickle.load(f_meta)
                    if prev_meta == meta:
                        split_paths.append(split_path)
                        pbar.update(1 + len(chunk))
                        continue

                new_header = {}
                data_start = 0
                data_blocks = []

                for name, info, _ in chunk:
                    start, end = info["data_offsets"]
                    size = end - start
                    data_blocks.append((name, start, size))
                    new_header[name] = {
                        "dtype": info["dtype"],
                        "shape": info["shape"],
                        "data_offsets": (data_start, data_start + size)
                    }
                    data_start += size

                # Сериализуем заголовок
                header_bytes = json.dumps(new_header).encode("utf-8")
                header_bytes_len = len(header_bytes)
                header_pad = -header_bytes_len & 7

                # Создаём файл
                with open(split_path, "wb") as f_out:
                    f_out.write(struct.pack("<Q", header_bytes_len + header_pad))
                    f_out.write(header_bytes)
                    f_out.write(b" " * header_pad) # b"\0" ломает json.loads, поэтому здесь b" "
                    pbar.update(1)

                    for _, offset, size in data_blocks:
                        f_in.seek(offset + header_offset)
                        buffer = f_in.read(size)
                        f_out.write(buffer)
                        f_out.flush()
                        del buffer
                        pbar.update(1)

                # Сохраняем .meta
                with open(meta_path, "wb") as f_meta:
                    pickle.dump(meta, f_meta)

                split_paths.append(split_path)

        return ({
            "clip_split_paths": split_paths,
            "clip_type": clip_type,
            "device": device
        },)



class CLIPTextEncode(ComfyNodeABC):
    @classmethod
    def INPUT_TYPES(s) -> InputTypeDict:
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "dynamicPrompts": True, "tooltip": "The text to be encoded."}),
                "meta clip": ("CLIP_META", {"tooltip": "The CLIP_META model used for encoding the text."})
            }
        }
    RETURN_TYPES = ("CONDITIONING",)
    OUTPUT_TOOLTIPS = ("A conditioning containing the embedded text used to guide the diffusion model.",)
    FUNCTION = "encode"

    CATEGORY = "VectorASD 🔥/AI"
    DESCRIPTION = """
Encodes a text prompt using a CLIP_META model into an embedding
that can be used to guide the diffusion model towards generating specific images.
It only works with CLIP_META in order to be accessible to people with any video card.
"""

    def encode(self, text, **kwargs):
        meta_clip = kwargs.get("meta clip", None)
        if meta_clip is None:
            raise RuntimeError("ERROR: clip input is invalid: None\n\nIf the clip is from a checkpoint loader node your checkpoint does not contain a valid clip or text encoder model.")

        clip_path     = meta_clip["clip_path"]
        clip_chunks   = meta_clip["clip_chunks"]
        clip_type     = meta_clip.get("clip_type", "stable_diffusion")
        device        = meta_clip.get("device",    "default")

        full_header = utils.chunks_to_full_header(clip_chunks)

        data = "TEXT", text

        for i, chunk in enumerate(clip_chunks, 1):
            print(f"🔄 Загружаем {i}-ую часть модели...")
            clip, parts = sd.load_clip_direct((clip_path,), full_header, chunk, clip_type, device)
            print(f"✅ {i}-ая часть загружена ({parts})")

            data = umt5_xxl.encode_from_tokens_scheduled(clip, data, parts=parts)
            print("  data TYPE:", data[0])

            del clip
            torch.cuda.empty_cache()
            print(f"🧹 {i}-ая часть выгружена из GPU")

        assert data[0] == "SCHEDULED", f"required type 'SCHEDULED', but arrived {data[0]}"
        TYPE, all_cond_pooled = data

        if False: # debug
            tensor, pooled_dict = all_cond_pooled[0]

            print("📤 Финальный тензор:")
            print(tensor)
            utils.visualize_tensor_jump(tensor)
            print("pooled_dict:", pooled_dict)
            digest = utils.tensor_sha256_digest(tensor)
            print("CHIEF CHECK!!!", digest.hex().startswith("3ca62203c0864acc"))

        return (all_cond_pooled, )



class CompareAnything:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "a": (IO.ANY, {"tooltip": "The first value to compare. Can be any type."}),
                "b": (IO.ANY, {"tooltip": "The second value to compare. Can be any type."}),
                "mode": (
                    ("==", "!=", "<", ">", "<=", ">="),
                    {"tooltip": "The comparison operator to apply between the two values."}
                ),
            }
        }

    RETURN_TYPES = ("BOOLEAN",)
    RETURN_NAMES = ("result",)
    OUTPUT_TOOLTIPS = ("The result of the comparison: True or False.",)
    FUNCTION = "compare"

    CATEGORY = "VectorASD 🔥/Logic"
    DESCRIPTION = """
Compares two values of any type using the selected comparison operator.
Supports deep comparison of nested structures such as lists, tuples, dictionaries, sets,
as well as NumPy arrays and PyTorch tensors. Uses a recursive comparator with cycle protection.
"""

    def compare(self, a, b, mode):
        comparator = utils.make_comparator(mode)
        return (comparator(a, b),)



NODE_MAPPINGS = {
    "ASD_CLIPLoader":      {"class": CLIPLoader,     "name": "Meta CLIP Loader"},
    "ASD_CLIPSplitter":    {"class": CLIPSplitter,   "name": "Meta CLIP Splitter"},
    "ASD_CLIPTextEncode":  {"class": CLIPTextEncode, "name": "Meta CLIP Text Encoder"},
    "ASD_CompareAnything": {"class": CompareAnything, "name": "Compare Anything"},
}
