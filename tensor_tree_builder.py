# Copyright 2025 VectorASD
# Licensed under the Apache License, Version 2.0 (see LICENSE file in project root).
#
# This code reconstructs and adapts logic originally found in a miniified JavaScript snippet
# used to build a tree structure from safetensors metadata in the Comfy-Org Hugging Face repository:
# https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/tree/main/split_files/text_encoders
#
# Translated to Python with the assistance of Microsoft Copilot and further modified by VectorASD.
# Note: this code does not render the tree as a table — it only constructs a nested structure from safetensors keys.
# Original license unknown. If you are the author, please contact to clarify attribution.

import re
from functools import cmp_to_key
from typing import List, Dict, Optional, Union

# Разделитель имени тензора
TENSOR_NAME_SPLITTER = re.compile(r"(\d+|\.)")
NAME_OFFSET = 6

class TensorEntry:
    def __init__(self, name: str, value: Optional[Dict] = None):
        self.name = name
        self.value = value
        self.name_parsed = self.parse_name(name)

    @staticmethod
    def parse_name(name: str) -> List[Union[str, int]]:
        return [int(part) if part.isdigit() else part for part in TENSOR_NAME_SPLITTER.split(name) if part]

    def __str__(self, indent: int = 0) -> str:
        return f"{' ' * indent}{self.name}\t{self.value.get('shape', [])}\t{self.value.get('dtype', '?')}"



class BaseTreeNode:
    def __init__(self, label: str, children: Dict[str, 'BaseTreeNode'], value: Optional[Dict] = None):
        self.label = label
        self.children = children
        self.value = value
        self.name_parsed = self.parse_name(label)

    def create_node(self, label: str, children: Dict[str, 'BaseTreeNode'], value: Optional[Dict] = None):
        raise NotImplementedError("Must be implemented in subclass")

    @property
    def is_leaf_node(self) -> bool:
        return self.value is not None

    @property
    def children_values(self) -> List['BaseTreeNode']:
        return list(self.children.values())

    def parse_name(self, name: str) -> List[Union[str, int]]:
        return [int(part) if part.isdigit() else part for part in TENSOR_NAME_SPLITTER.split(name) if part]

    def insert(self, tensor: TensorEntry):
        parts = []
        for part in tensor.name_parsed:
            if part == ".":
                last = parts.pop() if parts else ""
                parts.append(last + ".")
            else:
                parts.append(str(part))

        tail = "".join(parts[NAME_OFFSET:])
        head = parts[:NAME_OFFSET]
        if tail:
            head.append(tail)

        current = self
        path = ""
        for segment in head:
            path += segment
            if path not in current.children:
                current.children[path] = self.create_node(path, {})
            current = current.children[path]

        current.value = tensor.value

    def combine_single_child_parent(self):
        keys = list(self.children.keys())
        if len(keys) == 1:
            child = self.children[keys[0]]
            self.label = child.label
            self.value = child.value
            self.children = child.children
            self.combine_single_child_parent()
        else:
            for child in self.children.values():
                child.combine_single_child_parent()

    def __str__(self, indent: int = 0) -> str:
        pad = " " * indent
        if self.is_leaf_node:
            return f"{pad}{self.label}\t{self.value.get('shape', [])}\t{self.value.get('dtype', '?')}"
        else:
            count = len(self.children)
            return f"{pad}{self.label}({count})"



class TensorTreeNode(BaseTreeNode):
    def __init__(self, label: str, children: Dict[str, 'TensorTreeNode'], value: Optional[Dict] = None):
        super().__init__(label, children, value)
        self.n_params = 0
        self.percentage_params = 0

    def create_node(self, label: str, children: Dict[str, 'TensorTreeNode'], value: Optional[Dict] = None):
        return TensorTreeNode(label, children, value)

    def calculate_n_params(self) -> int:
        if self.value and "shape" in self.value:
            shape = self.value["shape"]
            # self.n_params = 0 if not shape else self._product(shape)
            self.n_params = self._product(shape)
        else:
            self.n_params = sum(child.calculate_n_params() for child in self.children_values)
        return self.n_params

    def calculate_percentage_params(self, total: int):
        self.percentage_params = self.n_params / total if total else 0
        for child in self.children_values:
            child.calculate_percentage_params(total)

    @staticmethod
    def _product(lst: List[int]) -> int:
        result = 1
        for x in lst:
            result *= x
        return result

    def __str__(self, indent: int = 0) -> str:
        first = super().__str__(indent)
        indent += 2
        return "\n".join((first, *(child.__str__(indent) for child in self.children.values())))

    def collect(self, append):
        if self.is_leaf_node:
            append((self.label, self.value, self.n_params))
        else:
            for child in self.children.values(): child.collect(append)



class TensorTreeBuilder:
    def __init__(self, root: TensorTreeNode):
        self.root = root

    def insert_nodes(self, tensors: List[TensorEntry], sort: bool = True):
        if sort:
            tensors.sort(key=cmp_to_key(self.tensor_name_comparator))
        for tensor in tensors:
            self.insert(tensor)

    def insert(self, tensor: TensorEntry):
        self.root.insert(tensor)

    def combine_single_child_parent(self):
        self.root.combine_single_child_parent()

    @staticmethod
    def tensor_name_comparator(a: TensorEntry, b: TensorEntry) -> int:
        embed_pattern = re.compile(r"(embe?d|wte|wpe|shared)", re.IGNORECASE)
        output_pattern = re.compile(r"(head|classifier|output)", re.IGNORECASE)

        l = a.name_parsed
        r = b.name_parsed

        for i, j in zip(l, r):
            if isinstance(i, int) and isinstance(j, int):
                if i != j:
                    return i - j
            elif isinstance(i, str) and isinstance(j, str):
                if embed_pattern.search(i) != embed_pattern.search(j):
                    return -1 if embed_pattern.search(i) else 1
                if output_pattern.search(i) != output_pattern.search(j):
                    return 1 if output_pattern.search(i) else -1
                cmp = (i > j) - (i < j)
                if cmp != 0:
                    return cmp
            else:
                return -1 if isinstance(i, int) else 1

        return len(l) - len(r)


def build_tensor_tree(tensors: List[TensorEntry]) -> TensorTreeBuilder:
    root = TensorTreeNode("", {})
    builder = TensorTreeBuilder(root)
    builder.insert_nodes(tensors, sort=True)
    builder.combine_single_child_parent()
    builder.root.calculate_n_params()
    builder.root.calculate_percentage_params(builder.root.n_params)
    return builder

def header_to_tree(header):
    tensors = [TensorEntry(name, value) for name, value in header.items()]
    builder = build_tensor_tree(tensors)
    return builder.root
