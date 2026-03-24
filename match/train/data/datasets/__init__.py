# encoding: utf-8

from .CRTrack import CRTrack
from .dataset_loader import ImageDataset

__factory = {
    'CRTrack': CRTrack,
}


def get_names():
    return __factory.keys()


def init_dataset(name, *args, **kwargs):
    if name not in __factory.keys():
        raise KeyError("Unknown datasets: {}".format(name))
    return __factory[name](*args, **kwargs)
