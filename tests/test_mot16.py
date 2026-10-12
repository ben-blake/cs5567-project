"""Ground-truth parsing."""

import textwrap

import pytest

from mot.data.mot16 import (
    CLASS_PEDESTRIAN,
    CLASS_STATIC_PERSON,
    Box,
    group_by_frame,
    group_by_track,
    parse_annotations,
)

# The row from the course slides, plus a non-target class and a zero-conf row.
GT_SAMPLE = textwrap.dedent(
    """\
    1,3,586,447,85,263,1,1,1
    1,4,100,200,50,150,1,1,0.8
    2,3,590,450,85,263,1,1,1
    2,9,300,300,40,120,1,7,1
    3,3,594,452,85,263,0,1,1
    """
)


@pytest.fixture
def gt_file(tmp_path):
    path = tmp_path / "gt.txt"
    path.write_text(GT_SAMPLE)
    return path


def test_parses_every_row(gt_file):
    assert len(parse_annotations(gt_file)) == 5


def test_converts_xywh_to_zero_indexed_xyxy(gt_file):
    first = parse_annotations(gt_file)[0]
    assert first.frame == 1
    assert first.track_id == 3
    # File is 1-indexed: left 586 -> x1 585, and width is preserved.
    assert first.box.as_xyxy() == (585.0, 446.0, 670.0, 709.0)
    assert first.box.width == 85.0
    assert first.box.height == 263.0


def test_is_target_excludes_non_pedestrian_and_zero_confidence(gt_file):
    annotations = parse_annotations(gt_file)
    by_id = {(a.frame, a.track_id): a for a in annotations}

    assert by_id[(1, 3)].is_target
    assert by_id[(1, 3)].class_id == CLASS_PEDESTRIAN
    # Static person: a person, but not scored as a target.
    assert by_id[(2, 9)].class_id == CLASS_STATIC_PERSON
    assert not by_id[(2, 9)].is_target
    # Confidence 0 means "ignore this box".
    assert not by_id[(3, 3)].is_target


def test_grouping(gt_file):
    annotations = parse_annotations(gt_file)
    assert sorted(group_by_frame(annotations)) == [1, 2, 3]
    assert len(group_by_frame(annotations)[1]) == 2

    track_3 = group_by_track(annotations)[3]
    assert [a.frame for a in track_3] == [1, 2, 3]


def test_blank_and_comment_lines_are_skipped(tmp_path):
    path = tmp_path / "gt.txt"
    path.write_text("# header\n\n1,1,10,10,20,20,1,1,1\n")
    assert len(parse_annotations(path)) == 1


def test_det_file_short_rows(tmp_path):
    """det.txt has -1 fillers and no visibility column."""
    path = tmp_path / "det.txt"
    path.write_text("1,-1,100,200,50,150,0.9,-1,-1,-1\n")
    detection = parse_annotations(path)[0]
    assert detection.track_id == -1
    assert detection.confidence == pytest.approx(0.9)
    assert detection.class_id == CLASS_PEDESTRIAN  # -1 filler normalised


def test_box_clip():
    assert Box(-5, -5, 50, 50).clip(40, 30).as_xyxy() == (0.0, 0.0, 40.0, 30.0)
