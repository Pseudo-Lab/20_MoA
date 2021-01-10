import os
import json
from typing import Dict
from pprint import pprint

import numpy as np
import pandas as pd

from configs.tabnet import config


class SearchOption:
    def __init__(self, opt, type_):
        type_ = type_.lower()
        assert pd.api.types.is_list_like(opt)
        assert type_ in ["choice", "range"]
        self._opt = np.array(opt)
        self._type = type_
        if self._type == "range":
            assert len(self._opt) == 2
            assert self._opt[0] < self._opt[1]

    @property
    def opt(self):
        return self._opt

    @property
    def type(self):
        return self._type


class RandomParams:
    def __init__(self):
        self._range_dict = {}
        self._counter = 0

    def __call__(self, conf, make_d_a_same=True):
        for k, search_range in self._range_dict.items():
            if k in ['name', 'seed', 'device', 'data_dir', 'log_dir', 'bkup_dir', 'sub_dir']:
                continue

            # check unknown keys
            if not hasattr(conf, k):
                raise AttributeError(f"type {type(conf)} does not have attribute {k}.")

            if search_range.type == "choice":
                new_value = np.random.choice(search_range.opt)
            elif search_range.type == "range":
                min_, max_ = search_range.opt
                new_value = np.random.random()  # 0 ~ 1
                new_value = new_value * (max_ - min_) + min_
            else:
                raise NotImplementedError

            if isinstance(new_value, np.integer):
                new_value = int(new_value)

            # inplace
            setattr(conf, k, new_value)

        if make_d_a_same:
            # n_a <- n_d
            setattr(conf, 'n_a', getattr(conf, 'n_d'))

        self.save_config(conf)
        self._counter += 1

    def save_config(self, conf: config):
        d = {k: getattr(conf, k) for k in dir(conf) if not k.startswith('__')}
        filename = f"selected_params_{self._counter:02d}.json"
        save_path = os.path.join(conf.log_dir, filename)
        with open(save_path, 'w') as fp:
            json.dump(d, fp)
        # check
        with open(save_path, 'r') as fp:
            pprint(json.load(fp))

    def set(self, param_name, opt, type_):
        self._range_dict[param_name] = SearchOption(opt, type_)


if __name__ == "__main__":
    rp = RandomParams()
    rp.set("n_d", [8, 16, 32], 'choice')

    c = config()
    for _ in range(3):
        rp(c)
