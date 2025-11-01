# Copyright 2025 VectorASD
# Licensed under the Apache License, Version 2.0 (see LICENSE file in project root).

# To test this script, place it in the root directory of your ComfyUI installation and run:
#     python _test.py
#
# This assumes that ComfyUI is properly installed and accessible from the current working directory.

import importlib
import sys
import os
from pprint import pprint

import torch



TEXT = "Пушистые мягкие кошки"
MODEL_NAME = "umt5_xxl_fp8_e4m3fn_scaled.safetensors"

def major_test(meta_clip):
    clip_path   = meta_clip["clip_path"]
    clip_chunks = meta_clip["clip_chunks"]
    clip_type   = meta_clip.get("clip_type", "stable_diffusion")
    device      = meta_clip.get("device", "default")



    def run_split_model_sequential():
        full_header = utils.chunks_to_full_header(clip_chunks)

        data = "TEXT", TEXT

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

        tensor, pooled_dict = all_cond_pooled[0]

        print("📤 Финальный тензор:")
        print(tensor)
        utils.visualize_tensor_jump(tensor)
        print("pooled_dict:", pooled_dict)
        digest = utils.tensor_sha256_digest(tensor)
        print("CHIEF CHECK!!!", digest.hex().startswith("3ca62203c0864acc"))



    def tokenize(clip, text):
        tokenizer = clip.tokenizer # comfy.text_encoders.wan.WanT5Tokenizer
        print("clip:",      tokenizer.clip)      # umt5xxl
        print("clip_name:", tokenizer.clip_name) # clip_umt5xxl
        tokenizer = getattr(tokenizer, tokenizer.clip) # comfy.text_encoders.wan.UMT5XXlTokenizer
        tokenizer = tokenizer.tokenizer # comfy.text_encoders.spiece_tokenizer.SPieceTokenizer
        tokenizer = tokenizer.tokenizer # sentencepiece.SentencePieceProcessor; proxy of <Swig Object of type 'sentencepiece::SentencePieceProcessor *'

        tokens = clip.tokenize(text)
        for model, batch in tokens.items():
            for tokenz in batch:
                size = len(tokenz)
                tokenz    = tuple((token, w) for token, w in tokenz if token) # token = 0 = <pad>
                input_ids = tuple((tokenizer.id_to_piece(token), w) for token, w in tokenz)
                print(f"TOKENS({model})(size: {size}):", tokenz)
                print("  input_ids:", input_ids)
        return tokens

    def run_full_model():
        import folder_paths

        print("🔄 Загружаем полную модель...")
        clip_path = folder_paths.get_full_path_or_raise("text_encoders", MODEL_NAME)
        full_header = utils.load_full_header((clip_path,))
        clip, parts = sd.load_clip_direct((clip_path,), full_header, None, "wan")
        print(f"✅ Модель загружена ({parts})")
        # input("Enter...")

        tensors = []
        for i in range(2):
            if i == 0:
                print(f"🧠 Кодируем текст: {TEXT}")
                tokens = tokenize(clip, TEXT)
                all_cond_pooled = clip.encode_from_tokens_scheduled(tokens)
            else:
                data = "TEXT", TEXT

                # data = umt5_xxl.encode_from_tokens_scheduled(clip, data, parts={"spiece_model", "shared", "encoder", "final_layer_norm"})
                data = umt5_xxl.encode_from_tokens_scheduled(clip, data, parts={"spiece_model"})
                data = umt5_xxl.encode_from_tokens_scheduled(clip, data, parts={"shared"})
                data = umt5_xxl.encode_from_tokens_scheduled(clip, data, parts={"encoder"})
                data = umt5_xxl.encode_from_tokens_scheduled(clip, data, parts={"final_layer_norm"})

                assert data[0] == "SCHEDULED", f"required type 'SCHEDULED', but arrived {data[0]}"
                TYPE, all_cond_pooled = data

            tensor, pooled_dict = all_cond_pooled[0]

            print("📤 Тензор:", tensor.shape, tensor.dtype)
            print(tensor)
            utils.visualize_tensor_jump(tensor)
            print("pooled_dict:", pooled_dict)
            utils.tensor_sha256_digest(tensor)
            tensors.append(tensor)

        print("CHIEF CHECK:", (tensors[0] == tensors[1]).all())

        del clip
        torch.cuda.empty_cache()
        print("🧹 Полная модель выгружена из GPU")



    from time import time
    T1 = time()
    run_full_model()
    T2 = time()
    # exit()
    run_split_model_sequential()
    T3 = time()
    print("TIME:", T2 - T1, T3 - T2) # 10.877089262008667 7.040693283081055 ;'-}
    # Вывод: я не только уменьшил нагрузка на VRAM до минимума, но и УСКОРИЛ скрипт! (файл подкачки практически не используется)



# Добавляем корень проекта в sys.path
sys.path.append(os.path.abspath("custom_nodes"))

# Импортируем как пакет
module = importlib.import_module("ComfyUI-VectorASD")

CLIPLoader      = module.NODE_CLASS_MAPPINGS["ASD_CLIPLoader"]
CLIPSplitter    = module.NODE_CLASS_MAPPINGS["ASD_CLIPSplitter"]
CLIPTextEncode  = module.NODE_CLASS_MAPPINGS["ASD_CLIPTextEncode"]
CompareAnything = module.NODE_CLASS_MAPPINGS["ASD_CompareAnything"]

umt5_xxl = module.umt5_xxl
sd       = module.sd

utils = module.utils



def check_deep_comparator():
    a = {"x": [1, 2], "y": torch.tensor([3.0])}
    b_arr = (
        {"x": [1, 2], "y": torch.tensor([3.0])},  # equal
        {"x": [1, 2], "y": torch.tensor([3.1])},  # tensor differs
        {"x": [1, 2], "z": torch.tensor([3.0])},  # key differs
        {"x": (1, 2), "y": torch.tensor([3.0])},  # list vs tuple
        {"x": [1, 3], "y": torch.tensor([3.0])},  # list content differs
        {"x": [1, 1], "y": torch.tensor([3.0])},  # list content differs
    )

    node = CompareAnything()

    for mode in ["==", "!=", "<", ">", ">=", "<=", "~"]:
        print(f"\n--- Testing mode: {mode} ---")
        for i, b in enumerate(b_arr, 1):
            result = utils.make_comparator(mode)(a, b)
            print(f"Case {i}: {result}", node.compare(a, b, mode))

    a = [[torch.tensor([[
         [ 0.0016, -0.0376, -0.0150, 0.0006, -0.0047, -0.0361],
         [ 0.0003, -0.0215, -0.0105, 0.0012,  0.0215,  0.0071],
         [ 0.0016, -0.0047, -0.0541, 0.0003, -0.0571,  0.0190],
         [ 0.0000, -0.0000, -0.0000, 0.0000,  0.0000, -0.0000],
         [ 0.0000, -0.0000, -0.0000, 0.0000,  0.0000, -0.0000],
         [ 0.0000, -0.0000, -0.0000, 0.0000,  0.0000, -0.0000]]]), {'pooled_output': None}]]
    b = [[torch.tensor([[
         [ 0.0016, -0.0376, -0.0150, 0.0006, -0.0047, -0.0361],
         [ 0.0003, -0.0215, -0.0105, 0.0012,  0.0215,  0.0071],
         [ 0.0016, -0.0047, -0.0541, 0.0003, -0.0571,  0.0190],
         [ 0.0000, -0.0000, -0.0000, 0.0000,  0.0000, -0.0000],
         [ 0.0000, -0.0000, -0.0000, 0.0000,  0.0000, -0.0000],
         [ 0.0000, -0.0000, -0.0000, 0.0000,  0.0000, -0.0000]]]), {'pooled_output': None}]]

    for i in range(2):
        print(f"\n--- Testing mode: special ---")
        for mode in ["==", "!=", "<", ">", ">=", "<=", "~"]:
            result = utils.make_comparator(mode)(a, b)
            print(f"Case {mode}: {result}", node.compare(a, b, mode))
        b[0][0][0][0][0] = 0.5

check_deep_comparator(); exit()



meta_clip = CLIPLoader().load_clip_meta("umt5_xxl_fp8_e4m3fn_scaled.safetensors", "wan", chunk_size_gb=1.0)[0]
pprint(meta_clip)

# clip_split = CLIPSplitter().save_chunks(**{"meta clip": meta_clip})[0]
# pprint(clip_split)

# major_test(meta_clip); exit()

CLIPTextEncode().encode(TEXT, **{"meta clip": meta_clip})

input("Press enter to exit...")
