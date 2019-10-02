import os
from ..include.serializer import Serializer
from collections import namedtuple


def test_serializer_dict():
    obj = {
        "one": "два", 
        "three": 4}
    filepath = "tmp"
    Serializer.save(obj, filepath)
    assert obj == Serializer.load(filepath)

def test_serializer_list():
    obj = [1, 2, 3]
    filepath = "tmp"
    Serializer.save(obj, filepath)
    assert obj == Serializer.load(filepath)
    os.remove(filepath)

def test_serializer_namedtuple_0():
    named_data = namedtuple("IDF", ["word_count", "doc_count"])
    obj = named_data(1, 2)
    filepath = "tmp"
    Serializer.save(obj, filepath)
    assert obj == named_data(*Serializer.load(filepath))
    os.remove(filepath)

def test_serializer_namedtuple_1():
    named_data = namedtuple("IDF", ["word_count", "doc_count"])
    obj = named_data({"one":2, "два":4}, 2)
    filepath = "tmp"
    Serializer.save(obj, filepath)
    assert obj == named_data(*Serializer.load(filepath))
    os.remove(filepath)
