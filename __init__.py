# Copyright 2025 VectorASD
# Licensed under the Apache License, Version 2.0

from .nodes_string import NODE_MAPPINGS as NODE_MAPPINGS_STRING
from .nodes_ai     import NODE_MAPPINGS as NODE_MAPPINGS_AI



def generate_node_mappings(node_configs):
    node_class_mappings = {}
    node_display_name_mappings = {}

    for node_config in node_configs:
        for node_name, node_info in node_config.items():
            node_class_mappings       [node_name] = node_info["class"]
            node_display_name_mappings[node_name] = node_info.get("name", node_info["class"].__name__)

    return node_class_mappings, node_display_name_mappings

NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS = generate_node_mappings((
    NODE_MAPPINGS_STRING,
    NODE_MAPPINGS_AI
))
WEB_DIRECTORY = "./js"

__all__ = ("NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY")

# for _test.py
from . import umt5_xxl
from . import utils
from . import sd
