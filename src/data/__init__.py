from .eurosat import load_eurosat, load_eurosat_test
from .cifar10 import load_cifar, load_cifar_test

__all__ = [
	"load_eurosat",
	"load_eurosat_test",
	"load_cifar",
	"load_cifar_test",

]
from .synthetic import load_synthetic, load_synthetic_test

from .spcm import load_spcm, load_spcm_test


def registry(train=True,args=None):
    from functools import partial
    from .vision import CLASSES,load_vision,load_vision_test
    result={'eurosat':load_eurosat if train else load_eurosat_test,
            'spcm':load_spcm if train else load_spcm_test,
            'synthetic':load_synthetic if train else load_synthetic_test}
    root=getattr(args,'data_root',None)
    for name in CLASSES:
        if train:
            result[name]=partial(load_vision,name,root=root,human_labels=getattr(args,'human_labels',None),
                                 human_key=getattr(args,'human_noise_key',None))
        else:result[name]=partial(load_vision_test,name,root=root)
    return result
