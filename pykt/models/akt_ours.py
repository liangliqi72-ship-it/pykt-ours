"""Backward-compatible wrapper for DUGP-KT.

Old scripts use model_name='akt_ours'.  The formal implementation now lives in
akt_dugp.py; this file keeps the old name working.
"""

from .akt_dugp import AKTOurs, AKTDUGP
