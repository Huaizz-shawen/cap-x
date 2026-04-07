from capx.utils.launch_utils import _extract_code


def test_extract_code_handles_none_response() -> None:
    assert _extract_code(None) == [""]
