# coding: utf-8

from itertools import chain, starmap
from collections import Counter

import torch
from torchtext.data import Dataset as TorchtextDataset
from torchtext.data import Example


def _join_dicts(*args):
    """
    Args:
        dictionaries with disjoint keys.

    Returns:
        a single dictionary that has the union of these keys.
    """

    return dict(chain(*[d.items() for d in args]))

class Dataset(TorchtextDataset):
    """Contain data and process it.

    A dataset is an object that accepts sequences of raw data (sentence pairs
    in the case of machine translation) and fields which describe how this
    raw data should be processed to produce tensors. When a dataset is
    instantiated, it applies the fields' preprocessing pipeline (but not
    the bit that numericalizes it or turns it into batch tensors) to the raw
    data, producing a list of :class:`torchtext.data.Example` objects.
    torchtext's iterators then know how to use these examples to make batches.

    Args:
        fields (dict[str, Field]): a dict with the structure
            returned by :func:`onmt.inputters.get_fields()`. Usually
            that means the dataset side, ``"src"`` or ``"tgt"``. Keys match
            the keys of items yielded by the ``readers``, while values
            are lists of (name, Field) pairs. An attribute with this
            name will be created for each :class:`torchtext.data.Example`
            object and its value will be the result of applying the Field
            to the data that matches the key. The advantage of having
            sequences of fields for each piece of raw input is that it allows
            the dataset to store multiple "views" of each input, which allows
            for easy implementation of token-level features, mixed word-
            and character-level models, and so on. (See also
            :class:`onmt.inputters.TextMultiField`.)
        readers (Iterable[onmt.inputters.DataReaderBase]): Reader objects
            for disk-to-dict. The yielded dicts are then processed
            according to ``fields``.
        data (Iterable[Tuple[str, Any]]): (name, ``data_arg``) pairs
            where ``data_arg`` is passed to the ``read()`` method of the
            reader in ``readers`` at that position. (See the reader object for
            details on the ``Any`` type.)
        dirs (Iterable[str or NoneType]): A list of directories where
            data is contained. See the reader object for more details.
        sort_key (Callable[[torchtext.data.Example], Any]): A function
            for determining the value on which data is sorted (i.e. length).
        filter_pred (Callable[[torchtext.data.Example], bool]): A function
            that accepts Example objects and returns a boolean value
            indicating whether to include that example in the dataset.


    """

    def __init__(self, fields, readers, data, dirs, sort_key,
                 filter_pred=None):
        self.sort_key = sort_key
        read_iters = [r.read(dat[1], dat[0], dir_) for r, dat, dir_
                      in zip(readers, data, dirs)]


        examples = []
        for ex_dict in starmap(_join_dicts, zip(*read_iters)):
            ex_fields = {k: [(k, v)] for k, v in fields.items() if
                         k in ex_dict}
            ex = Example.fromdict(ex_dict, ex_fields)
            examples.append(ex)

        # fields needs to have only keys that examples have as attrs
        fields = []
        for _, nf_list in ex_fields.items():
            assert len(nf_list) == 1
            fields.append(nf_list[0])

        super(Dataset, self).__init__(examples, fields, filter_pred)

    def __getattr__(self, attr):
        # avoid infinite recursion when fields isn't defined
        if 'fields' not in vars(self):
            raise AttributeError
        if attr in self.fields:
            return (getattr(x, attr) for x in self.examples)
        else:
            raise AttributeError

    def save(self, path, remove_fields=True):
        if remove_fields:
            self.fields = []
        torch.save(self, path)
