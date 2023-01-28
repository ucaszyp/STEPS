import mmcv
from mmcv.utils import Registry


def _build_func(name: str, option: mmcv.ConfigDict, registry: Registry):
    return registry.get(name)(option)


MODELS = Registry('models', build_func=_build_func)
