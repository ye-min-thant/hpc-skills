"""Unit tests for ofdft-opt (stdlib unittest only).

Run from plugins/ofdft/package:
    python -m unittest discover -s tests -v
Each test module puts ../src on sys.path itself, so PYTHONPATH=src is optional.
DFTpy tests (test_dftpy_small.py) are skipped when dftpy or the BLPS files are unavailable.
"""
