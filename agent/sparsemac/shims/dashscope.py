"""Minimal dashscope shim for Wan generate.py when prompt extension is disabled.

The current vbench environment has Wan/diffusers but not dashscope. Wan imports
dashscope at module import time even when prompt extension is not used. This
shim is placed on PYTHONPATH only for the final full-generation subprocess.
"""


class MultiModalConversation:
    class Models:
        qwen_vl_max = "qwen-vl-max"

    @staticmethod
    def call(*args, **kwargs):
        raise RuntimeError("dashscope shim cannot run prompt extension")


class Generation:
    @staticmethod
    def call(*args, **kwargs):
        raise RuntimeError("dashscope shim cannot run prompt extension")

