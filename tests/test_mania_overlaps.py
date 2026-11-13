from osuT5.osuT5.inference.postprocessor import fix_mania_column_overlaps

BEAT = 400.0  # 150 BPM


def note(column, time, keys=4):
    return f"{int((column + 0.5) * 512 / keys)},192,{time},1,0,0:0:-1:0:"


def hold(column, time, end, keys=4):
    return f"{int((column + 0.5) * 512 / keys)},192,{time},128,0,{end}:0:0:-1:0:"


def fix(objects):
    return fix_mania_column_overlaps(objects, 4, lambda t: BEAT)


def test_valid_map_is_unchanged():
    objects = [note(0, 1000), hold(1, 1000, 1800), note(1, 2000), note(2, 1400)]
    fixed, fixes = fix(objects)
    assert fixes == 0
    assert sorted(fixed) == sorted(objects)


def test_hold_ending_on_next_note_is_shortened_by_a_quarter_beat():
    fixed, fixes = fix([hold(1, 40745, 41488), hold(1, 41488, 42158)])
    assert fixes == 1
    assert fixed == [hold(1, 40745, 41388), hold(1, 41488, 42158)]


def test_note_inside_hold_shortens_the_hold():
    fixed, fixes = fix([hold(2, 1000, 3000), note(2, 2000)])
    assert fixes == 1
    assert fixed == [hold(2, 1000, 1900), note(2, 2000)]


def test_too_short_hold_becomes_a_note():
    # Shortening to 1100 - 100 = 1000 would leave no hold at all
    fixed, fixes = fix([hold(0, 1000, 1500), note(0, 1100)])
    assert fixes == 1
    assert fixed == [note(0, 1000), note(0, 1100)]


def test_stacked_objects_keep_the_hold():
    fixed, fixes = fix([note(3, 1000), hold(3, 1000, 1600)])
    assert fixes == 1
    assert fixed == [hold(3, 1000, 1600)]


def test_other_columns_are_independent():
    objects = [hold(0, 1000, 2000), note(1, 1500), note(2, 1500), note(3, 1999)]
    fixed, fixes = fix(objects)
    assert fixes == 0
    assert sorted(fixed) == sorted(objects)


def test_output_is_sorted_by_time():
    fixed, _ = fix([note(3, 3000), note(0, 1000), note(2, 2000)])
    assert [int(s.split(",")[2]) for s in fixed] == [1000, 2000, 3000]
