# Copyright 2025 VectorASD
# Licensed under the Apache License, Version 2.0

from comfy.comfy_types.node_typing import IO

# my first custom nodes ;'-}



class JoinString:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "any1": (IO.ANY, {}), # позже узнал, что IO.ANY = "*", не "ANY"
            },
            "optional": {
                "any2": (IO.ANY, {}),
            }
        }
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("string",)
    FUNCTION = "joiner"
    CATEGORY = "VectorASD 🔥/string"
    DESCRIPTION = """
Concat two strings.
Supports all types of.
"""

    def joiner(self, any1="", any2=""):
        joined_string = f"{any1}{any2}"
        return (joined_string, )



class JoinStringPrefix:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prefix1": (IO.STRING, {"default": 'any1', "multiline": False}),
                "any1": (IO.ANY, {}),
            },
            "optional": {
                "prefix2": (IO.STRING, {"default": 'any2', "multiline": False}),
                "any2": (IO.ANY, {}),
            }
        }
    RETURN_TYPES = (IO.STRING,)
    RETURN_NAMES = ("string",)
    FUNCTION = "joiner"
    CATEGORY = "VectorASD 🔥/string"
    DESCRIPTION = """
Concat two strings with prefix.
Supports all types of.
The second string starts from a new line.
"""

    def joiner(self, prefix1="any1", any1="", prefix2="any2", any2=""):
        joined_string = f"{prefix1}: {any1}\n{prefix2}: {any2}"
        return (joined_string, )



class JoinStringMulti:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "any1": (IO.ANY, {}),
                "inputcount": (IO.INT, {"default": 2, "min": 2, "max": 1000, "step": 1}),
            },
            "optional": {
                "any2": (IO.ANY, {}),
            }
        }

    RETURN_TYPES = (IO.STRING,)
    RETURN_NAMES = ("string",)
    FUNCTION = "combine"
    CATEGORY = "VectorASD 🔥/string"
    DESCRIPTION = """
Concat any number of strings.
Supports all types of.
Click 'Update inputs' for apply 'inputcount'.
"""

    def combine(self, inputcount, any1="", **kwargs):
        result = (str(any1), *(str(kwargs.get(f"any{id}", "")) for id in range(2, inputcount + 1)))
        return ("".join(result),)



class JoinStringMultiPrefix:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "any1": (IO.ANY, {}),
                "inputcount": (IO.INT, {"default": 2, "min": 2, "max": 1000, "step": 1}),
                "prefix1": (IO.STRING, {"default": 'any1', "multiline": False}),
            },
            "optional": {
                "prefix2": (IO.STRING, {"default": 'any2', "multiline": False}),
                "any2": (IO.ANY, {}),
            }
        }

    RETURN_TYPES = (IO.STRING,)
    RETURN_NAMES = ("string",)
    FUNCTION = "combine"
    CATEGORY = "VectorASD 🔥/string"
    DESCRIPTION = """
Concat any number of strings.
Supports all types of.
Click 'Update inputs' for apply 'inputcount'.
WARNING! I still haven't been able to make inputWidgets appear normally.
"""

    def combine(self, inputcount, prefix1="any1", any1="", **kwargs):
        result = (f"{prefix1}: {any1}", *(f"{kwargs.get(f'prefix{id}', '')}: {kwargs.get(f'any{id}', '')}" for id in range(2, inputcount + 1)))
        return ("\n".join(result),)



NODE_MAPPINGS = {
    "ASD_JoinString":             {"class": JoinString,             "name": "Join String"},
    "ASD_JoinStringPrefix":       {"class": JoinStringPrefix,       "name": "Join String Prefix"},
    "ASD_JoinStringMulti":        {"class": JoinStringMulti,        "name": "Join String Multi"},
    "ASD_JoinStringMultiPrefix":  {"class": JoinStringMultiPrefix,  "name": "Join String Multi Prefix"},
}
