# 仅用于提前发现导入链上缺的包：内存里伪造 flash_attn（含任意子模块），再跑 train --dryrun
import importlib.abc
import importlib.machinery
import runpy
import sys
import types


class _Stub(types.ModuleType):
    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return lambda *a, **k: None


class _Finder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def find_spec(self, name, path, target=None):
        if name == "flash_attn" or name.startswith("flash_attn."):
            return importlib.machinery.ModuleSpec(name, self, is_package=True)
        return None

    def create_module(self, spec):
        m = _Stub(spec.name)
        m.__path__ = []
        m.__version__ = "2.7.3"
        return m

    def exec_module(self, module):
        pass


sys.meta_path.insert(0, _Finder())
import transformers.utils  # noqa: E402
import transformers.utils.import_utils as iu  # noqa: E402

iu.is_flash_attn_2_available = transformers.utils.is_flash_attn_2_available = lambda: True
sys.argv = ["train"] + sys.argv[1:]
runpy.run_module("cosmos_policy.scripts.train", run_name="__main__")
