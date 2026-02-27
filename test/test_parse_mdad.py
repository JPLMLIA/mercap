import pytest

from mercap.parse_mdad_v3 import parse_mem_id_to_mask_value


@pytest.mark.parametrize("mem_id, expected", [
    ("D01_031", 31),
    ("D16_001b", 134),
    ("D01_031+032", 31),
    ("G13_001b+002+017", 134),
])
def test_parse_mem_id_to_mask_value(mem_id, expected):
    result = parse_mem_id_to_mask_value(mem_id)
    assert result == expected
