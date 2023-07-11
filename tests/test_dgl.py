import importlib
import sys

import pytest

from libreco.graph import check_dgl


def test_dgl(prepare_feat_data, monkeypatch):
    *_, data_info = prepare_feat_data

    with monkeypatch.context() as m:
        m.setitem(sys.modules, "dgl", None)
        with pytest.raises(ModuleNotFoundError):
            from libreco.algorithms import PinSageDGL

            _ = PinSageDGL("ranking", data_info)

    @check_dgl
    class ClsWithDGL:
        def __new__(cls, *args, **kwargs):
            if cls.dgl_error is not None:
                raise cls.dgl_error
            cls._dgl = importlib.import_module("dgl")
            return super().__new__(cls)

    model = ClsWithDGL()
    assert model.dgl_error is None
    assert model._dgl.__name__ == "dgl"
