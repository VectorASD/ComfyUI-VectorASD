# Copyright 2025 VectorASD
# Licensed under the Apache License, Version 2.0 (see LICENSE file in project root).

import struct, json
from math import pi, cos, sin, exp
from hashlib import sha256

import numpy as np
import torch
from safetensors import torch as safe_torch

name_to_type = safe_torch._TYPES
type_to_name = {v: k for k, v in name_to_type.items()}
name_to_size = {name: _type.itemsize for name, _type in name_to_type.items()}
name_to_size['?'] = 0



def load_header(clip_path):
    with open(clip_path, "rb") as file:
        header_len = struct.unpack("<Q", file.read(8))[0]
        json_data = file.read(header_len)
    header = json.loads(json_data)
    return header, header_len

class FakeTorch:
    def __init__(self, dtype, shape):
        self.dtype = name_to_type[dtype]
        self.shape = shape
    def __repr__(self):
        return f"{type(self).__name__}({type_to_name[self.dtype]}{self.shape})"

def load_full_header(split_paths):
    items = {}
    for path in split_paths:
        items.update(load_header(path)[0])
    full_header = {
        key: FakeTorch(value["dtype"], value["shape"])
        for key, value in items.items()}
    return full_header

def chunks_to_full_header(chunks):
    full_header = {
        key: FakeTorch(value["dtype"], value["shape"])
        for chunk in chunks
        for key, value, _ in chunk
    }
    return full_header



def visualize_embeddings(tensor, line_width=128, threshold=1e-6):
    batch_size, seq_len, hidden_dim = tensor.shape
    symbols = (' ', '·', '░', '▒', '▓', '█')
    chunk_size = (hidden_dim + line_width - 1) // line_width  # округление вверх

    for b in range(batch_size):
        for e in range(seq_len):
            emb = tensor[b, e]

            if torch.all(torch.abs(emb) < threshold): continue

            prefix = f"b{b}:e{e}"
            line = []

            for i in range(line_width):
                start = i * chunk_size
                end = min((i + 1) * chunk_size, hidden_dim)
                chunk = emb[start:end]

                val = torch.sum(torch.abs(chunk)).item() % 1.0
                idx = int(val * (len(symbols) - 1))
                line.append(symbols[idx])

            print(prefix, ''.join(line))

def visualize_tensor_jump(tensor, height=24, scale=256, threshold=1e-6):
    symbols = (' ', '.', '·', ':', '-', '°', '+', 'o', '*', '=', '•', '░', 'O', '▒', '●', '#', '■', '▓', '%', '@', '█')
    width = height * 2
    canvas = np.zeros((height, width), dtype=int)

    x = y = 0

    batch_size, seq_len, hidden_dim = tensor.shape

    for b in range(batch_size):
        for e in range(seq_len):
            emb = tensor[b, e]

            if torch.all(torch.abs(emb) < threshold):
                continue

            for w in emb:
                # Балансировщик чувствительности
                # angle = sin((1 - abs(w.item())) * pi) * pi
                angle = (1 - abs(w.item())) ** 2 * pi * 2 # Альтернатива с более резким откликом

                x += cos(angle) * 2
                y += sin(angle)

                xi = int(round(x / scale + width  / 2)) % width
                yi = int(round(y / scale + height / 2)) % height

                canvas[yi][xi] += 1

    max_val = np.max(canvas)
    num_symbols = len(symbols)
    for row in canvas:
        # line = ''.join(symbols[val % len(symbols)] for val in row) циклический 
        line = ''.join(
            symbols[min(round(val / max_val * (num_symbols - 1)), num_symbols - 1)] if max_val > 0 else symbols[0]
            for val in row
        )
        print("|", line, "|")

    # height, scale = map(int, input("height and scale: ").split())
    # visualize_tensor_jump(tensor, height, scale, threshold)

def tensor_sha256_digest(tensor: torch.Tensor) -> str:
    device = tensor.device
    if device.type != "cpu": tensor = tensor.contiguous().cpu()
    digest = sha256(tensor.numpy().tobytes()).digest()

    print(f"tensor ({device}) ({type_to_name[tensor.dtype]}{list(tensor.shape)}) sha256: {digest.hex()[:16]}...")

    return digest



def make_comparator(mode="=="):
    visited_lists = set()
    visited_tuples = set()
    visited_sets = set()
    visited_dicts = set()

    def get_apply_mode(mode):
        match mode:
            case "==": return lambda a, b: bool(a == b)
            case "!=": return lambda a, b: bool(a == b)
            case "<":  return lambda a, b: bool(a < b)
            case ">":  return lambda a, b: bool(a > b)
            case "<=": return lambda a, b: bool(a <= b)
            case ">=": return lambda a, b: bool(a >= b)
            case _:    return lambda a, b: False

    apply_mode = get_apply_mode(mode)
    equal_mode = mode in ("==",  "!=")

    def deep_compare(a, b):
        if type(a) != type(b):
            return mode == "!="

        if isinstance(a, (int, float, str, bool, type(None))):
            return apply_mode(a, b)

        if isinstance(a, np.ndarray):
            if equal_mode:
                return np.array_equal(a, b)
            try:
                a_val = np.all(a).item() if hasattr(np.all(a), 'item') else bool(np.all(a))
                b_val = np.all(b).item() if hasattr(np.all(b), 'item') else bool(np.all(b))
                return apply_mode(a_val, b_val)
            except:
                return False

        if isinstance(a, torch.Tensor):
            if equal_mode:
                return torch.equal(a, b)
            try:
                a_val = torch.all(a).item()
                b_val = torch.all(b).item()
                return apply_mode(a_val, b_val)
            except:
                return False

        if isinstance(a, list):
            aid, bid = id(a), id(b)
            if aid in visited_lists or bid in visited_lists:
                return True
            visited_lists.update([aid, bid])
            if len(a) != len(b): return False
            if equal_mode: return all(deep_compare(x, y) for x, y in zip(a, b))
            try: return apply_mode(a, b)
            except: return False

        if isinstance(a, tuple):
            aid, bid = id(a), id(b)
            if aid in visited_tuples or bid in visited_tuples:
                return True
            visited_tuples.update([aid, bid])
            if len(a) != len(b): return False
            if equal_mode: return all(deep_compare(x, y) for x, y in zip(a, b))
            try: return apply_mode(a, b)
            except: return False

        if isinstance(a, set):
            aid, bid = id(a), id(b)
            if aid in visited_sets or bid in visited_sets:
                return True
            visited_sets.update([aid, bid])
            try:
                return apply_mode(a, b)
            except:
                return False

        if isinstance(a, dict):
            aid, bid = id(a), id(b)
            if aid in visited_dicts or bid in visited_dicts:
                return True
            visited_dicts.update([aid, bid])
            if equal_mode and (len(a) != len(b) or a.keys() != b.keys()): return False
            # Сравниваем пары значений по ключам
            a_items = tuple((k, a[k]) for k in sorted(a))
            b_items = tuple((k, b[k]) for k in sorted(b))
            try:
                return apply_mode(a_items, b_items)
            except:
                return False

        if hasattr(a, '__dict__') and hasattr(b, '__dict__'):
            return deep_compare(a.__dict__, b.__dict__)

        if hasattr(a, '__slots__') and hasattr(b, '__slots__'):
            for slot in a.__slots__:
                if not deep_compare(getattr(a, slot), getattr(b, slot)):
                    return mode == "!="
            return mode == "=="

        return apply_mode(a, b)

    def final_compare(a, b):
        result = deep_compare(a, b)
        return not result if mode == "!=" else result

    return final_compare
