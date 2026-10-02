from callcheck import hand_labels
from callcheck.hand_labels import HAND_LABELS, hand_judge


def test_docstring_says_not_model_output():
    assert "NOT model output" in hand_labels.__doc__


def test_every_thread_labelled(threads):
    assert {t.thread_id for t in threads} == set(HAND_LABELS)


def test_hand_judge_source_is_hand(spec, threads):
    j = hand_judge().judge(spec, threads[0])
    assert j is not None and j.source == "hand" and j.model == "hand-labels"
    assert set(j.items) == set(spec.items)
