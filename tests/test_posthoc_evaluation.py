from argparse import Namespace

import pytest

from evaluate import parse_resa_energy, parse_resa_stream_weights, validate_posthoc_args


def _args(**overrides):
    values = {
        "resa_energy": None,
        "resa_cauchy_clip": None,
        "resa_stream_weights": None,
        "eval_dyn_groups": None,
        "reference_path": None,
        "eval_batch_size": None,
        "result_tag": None,
    }
    values.update(overrides)
    return Namespace(**values)


def test_parse_resa_energy_single_and_multi_stream():
    assert parse_resa_energy("u1") == "u1"
    assert parse_resa_energy("u1,u2,rec") == ["u1", "u2", "rec"]


def test_parse_resa_energy_rejects_unknown_stream():
    with pytest.raises(ValueError, match="Unknown"):
        parse_resa_energy("u1,bad")


def test_parse_resa_stream_weights_normalizes_values():
    assert parse_resa_stream_weights("2,8") == pytest.approx([0.2, 0.8])


@pytest.mark.parametrize("value", ["", "0,0", "-1,2", "nan,1"])
def test_parse_resa_stream_weights_rejects_invalid_values(value):
    with pytest.raises(ValueError, match="stream_weights"):
        parse_resa_stream_weights(value)


def test_posthoc_override_requires_result_tag():
    with pytest.raises(ValueError, match="result_tag"):
        validate_posthoc_args(_args(resa_energy="u1"))


def test_posthoc_args_accept_safe_independent_output():
    validate_posthoc_args(
        _args(
            resa_energy="u1,u2",
            resa_cauchy_clip=1.0,
            result_tag="rea_clip1",
        )
    )


def test_eval_batch_size_must_be_positive():
    with pytest.raises(ValueError, match="eval_batch_size"):
        validate_posthoc_args(_args(eval_batch_size=0))
    validate_posthoc_args(_args(eval_batch_size=1))


def test_eval_groups_requires_alternate_reference_path():
    with pytest.raises(ValueError, match="reference_path"):
        validate_posthoc_args(_args(eval_dyn_groups=2, result_tag="ctm_groups2"))

    validate_posthoc_args(
        _args(
            eval_dyn_groups=2,
            reference_path="posthoc_refs/ctm_groups2.npz",
            result_tag="ctm_groups2",
        )
    )


@pytest.mark.parametrize("tag", ["../escape", "BadTag", "bad tag", "_prefix"])
def test_posthoc_args_reject_unsafe_result_tag(tag):
    with pytest.raises(ValueError, match="result_tag"):
        validate_posthoc_args(_args(result_tag=tag))
