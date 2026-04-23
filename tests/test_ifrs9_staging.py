"""Unit tests on the staging logic of ecl.engine.assign_stage."""

from __future__ import annotations

import numpy as np
import pytest

from ecl.engine import assign_stage


def test_no_sicr_gives_stage1():
    stages = assign_stage(
        pd_current=np.array([0.01, 0.05, 0.10]),
        pd_origination=np.array([0.01, 0.05, 0.10]),
        days_past_due=np.array([0, 0, 0]),
        default_flag_at_asof=np.array([0, 0, 0]),
    )
    assert stages.tolist() == [1, 1, 1]


def test_dpd30_triggers_stage2():
    stages = assign_stage(
        pd_current=np.array([0.05]),
        pd_origination=np.array([0.05]),
        days_past_due=np.array([35]),
        default_flag_at_asof=np.array([0]),
    )
    assert stages.tolist() == [2]


def test_pd_doubling_triggers_stage2():
    stages = assign_stage(
        pd_current=np.array([0.15]),
        pd_origination=np.array([0.05]),    # 3x -> SICR
        days_past_due=np.array([0]),
        default_flag_at_asof=np.array([0]),
    )
    assert stages.tolist() == [2]


def test_dpd90_triggers_stage3():
    stages = assign_stage(
        pd_current=np.array([0.02]),
        pd_origination=np.array([0.02]),
        days_past_due=np.array([95]),
        default_flag_at_asof=np.array([0]),
    )
    assert stages.tolist() == [3]


def test_default_flag_triggers_stage3():
    stages = assign_stage(
        pd_current=np.array([0.02]),
        pd_origination=np.array([0.02]),
        days_past_due=np.array([0]),
        default_flag_at_asof=np.array([1]),
    )
    assert stages.tolist() == [3]


def test_zero_origination_pd_does_not_crash():
    """PD_origination=0 should not produce div-by-zero / NaN crashes."""
    stages = assign_stage(
        pd_current=np.array([0.05]),
        pd_origination=np.array([0.0]),
        days_past_due=np.array([0]),
        default_flag_at_asof=np.array([0]),
    )
    assert stages.tolist()[0] in (1, 2)
