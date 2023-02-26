#!/usr/bin/env python3

import copy
import torch


class DynamicalState(object):
    """Plain python object that models a dynamical state z and possibly
    manages other variables
    """

    def __init__(self, state, **kwargs):
        self.state = state

        for key, item in kwargs.items():
            self[key] = item

    @property
    def keys(self):
        r"""Returns all names of graph attributes."""
        keys = [key for key in self.__dict__.keys() if self[key] is not None]
        keys = [key for key in keys if key[:2] != '__' and key[-2:] != '__']
        return keys

    def clone(self):
        obj = DynamicalState(None)

        for key in self.__dict__.keys():
            if self[key] is not None:
                obj[key] = self[key].clone()

        return obj

    def __getitem__(self, key):
        r"""Gets the data of the attribute :obj:`key`."""
        return getattr(self, key, None)

    def __setitem__(self, key, value):
        """Sets the attribute :obj:`key` to :obj:`value`."""
        setattr(self, key, value)

    def __len__(self):
        """Returns the number of all present attributes."""
        return len(self.keys)

    def __contains__(self, key):
        """Returns :obj:`True`, if the attribute :obj:`key` is present in the
        data."""
        return key in self.keys

    def rename(self, key_dict):
        assert isinstance(key_dict, dict)

        for key_old, new_key in key_dict.items():
            self[key_new] = self.__dict__.pop(key_old)

    def append_or_create(self, key, value):
        if hasattr(self, key):
            self[key] = torch.cat([self[key], value])
        else: self[key] = value

    def add_or_create(self, key, value):
        if hasattr(self, key):
            self[key] += value
        else: self[key] = value

    def remove_key(self, key):
        del self.__dict__[key]

    def remove_all_keys_but(self, exception_keys):
        if isinstance(exception_keys, list):
            exception_keys = [exception_keys]

        key_list_cp = copy.copy(list(self.__dict__.keys()))

        for key in key_list_cp:
            if key not in exception_keys:
                self.remove_key(self,key)

    def __str__(self):
        repr = ""
        n_keys = len(self.__dict__.keys())
        for i, k in enumerate(self.__dict__.keys()):
            repr += str(k) + '=' + str([*self[k].size()])
            if i < n_keys -1: repr += ','

        return "DynamicState({})".format(repr)

    def __repr__(self):
        return str(self)
