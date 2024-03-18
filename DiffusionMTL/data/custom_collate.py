# This code is referenced from 
# https://github.com/facebookresearch/astmt/
# 
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# 
# License: Attribution-NonCommercial 4.0 International
import pdb
import torch
import collections
import re
int_classes = int
string_classes = str

_use_shared_memory = False
r"""Whether to use shared memory in default_collate"""


numpy_type_map = {
    'float64': torch.DoubleTensor,
    'float32': torch.FloatTensor,
    'float16': torch.HalfTensor,
    'int64': torch.LongTensor,
    'int32': torch.IntTensor,
    'int16': torch.ShortTensor,
    'int8': torch.CharTensor,
    'uint8': torch.ByteTensor,
}


def collate_mil(batch):
    """
    Puts each data field into a tensor with outer dimension batch size.
    Custom-made for supporting MIL
    """
    error_msg = "batch must contain tensors, numbers, dicts or lists; found {}"
    elem_type = type(batch[0])
    if isinstance(batch[0], torch.Tensor):
        out = None
        if _use_shared_memory:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in batch])
            storage = batch[0].storage()._new_shared(numel)
            out = batch[0].new(storage)
        return torch.stack(batch, 0, out=out)

    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        elem = batch[0]
        if elem_type.__name__ == 'ndarray':
            # array of string classes and object
            if re.search('[SaUO]', elem.dtype.str) is not None:
                raise TypeError(error_msg.format(elem.dtype))

            return torch.stack([torch.from_numpy(b) for b in batch], 0)
        if elem.shape == ():  # scalars
            py_type = float if elem.dtype.name.startswith('float') else int
            return numpy_type_map[elem.dtype.name](list(map(py_type, batch)))

    elif isinstance(batch[0], int_classes):
        return torch.LongTensor(batch)

    elif isinstance(batch[0], float):
        return torch.DoubleTensor(batch)

    elif isinstance(batch[0], string_classes):
        return batch

    elif isinstance(batch[0], collections.Mapping):
        batch_modified = {key: collate_mil([d[key] for d in batch]) for key in batch[0] if key.find('idx') < 0}
        if 'edgeidx' in batch[0]:
            batch_modified['edgeidx'] = [batch[x]['edgeidx'] for x in range(len(batch))]
        return batch_modified

    elif isinstance(batch[0], collections.Sequence):
        # transposed = zip(*batch)
        # return [collate_mil(samples) for samples in transposed]

        # yhr: change this for tolerating lists with different element number from different samples.
        out = []
        for samples in batch:
            out.append(collate_mil(samples))
        return out


    raise TypeError((error_msg.format(type(batch[0]))))
