"""Smoke tests for the CLI and example/package importability."""

import importlib

import pytest


def test_cli_info_runs(capsys):
    import sys

    from tessera.cli import main

    argv = sys.argv
    sys.argv = ["tessera", "info"]
    try:
        main()
    finally:
        sys.argv = argv
    out = capsys.readouterr().out
    assert "tessera-tiny" in out
    assert "params" in out


@pytest.mark.parametrize(
    "module",
    [
        "tessera",
        "tessera.model",
        "tessera.kernels",
        "tessera.quant",
        "tessera.serve",
        "tessera.distill",
        "tessera.interp",
        "tessera.data",
        "tessera.profiling",
    ],
)
def test_subpackages_import(module):
    assert importlib.import_module(module) is not None
