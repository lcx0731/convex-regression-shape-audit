import json

import pytest

from convex_shape.cli import main


def test_cli_valid_input_unchanged(tmp_path, capsys):
    path = tmp_path/'input.json'
    original = '{"x":[0,1,4],"y":[0,2,5],"predict_x":[0,2,4]}'
    path.write_text(original)
    assert main([str(path)]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["status"] == "converged"
    assert len(out["predictions"]) == 3
    assert path.read_text() == original


@pytest.mark.parametrize("input_text", ["{}", "[]", "invalid", '{"x":[],"x":[],"y":[]}', '{"x":[0,1],"y":[0,1],"extra":1}', '{"x":[0,1],"y":[0,1],"weights":null}', '{"x":[0,1],"y":[0,1],"predict_x":[2]}', " "*(1024*1024+1)], ids=["empty", "array", "syntax", "duplicate", "unknown", "null", "extrapolate", "oversize"])
def test_invalid_cli(tmp_path, capsys, input_text):
    path = tmp_path/'input.json'
    path.write_text(input_text)
    assert main([str(path)]) == 1
    assert capsys.readouterr().err.startswith('convex-fit:')


def test_cli_budget_exit(tmp_path, capsys):
    path = tmp_path/'input.json'
    path.write_text(json.dumps({"x": list(range(8)), "y": [0,4,0,4,0,4,0,4], "max_steps": 1}))
    assert main([str(path)]) == 3
    assert json.loads(capsys.readouterr().out)["status"] == "iteration_limit"


def test_missing_file(tmp_path, capsys):
    assert main([str(tmp_path/'missing')]) == 1
    assert 'convex-fit:' in capsys.readouterr().err
