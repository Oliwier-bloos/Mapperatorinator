from osuT5.osuT5.inference.postprocessor import full_chord_drop_order, limit_mania_full_chord_runs


def column_x(column, keys=4):
    return int((column + 0.5) * 512 / keys)


def note(column, time, keys=4):
    return f"{column_x(column, keys)},192,{time},1,0,0:0:-1:0:"


def hold(column, time, end, keys=4):
    return f"{column_x(column, keys)},192,{time},128,0,{end}:0:0:-1:0:"


def full_chord(time, keys=4):
    return [note(c, time, keys) for c in range(keys)]


def chords_of(objects, keys=4):
    chords = {}
    for s in objects:
        parts = s.split(",")
        chords.setdefault(int(parts[2]), set()).add(int(float(parts[0]) * keys / 512))
    return [chords[t] for t in sorted(chords)]


def test_drop_order_alternates_outer_columns():
    assert full_chord_drop_order(4) == [3, 0, 2, 1]
    assert full_chord_drop_order(7) == [6, 0, 5, 1, 4, 2, 3]


def test_off_by_default():
    objects = [s for t in range(0, 900, 150) for s in full_chord(t)]
    assert limit_mania_full_chord_runs(objects, 4, 0) == (objects, 0)


def test_quad_train_becomes_split_chordjack():
    objects = [s for t in range(0, 900, 150) for s in full_chord(t)]  # 6 quads
    result, removed = limit_mania_full_chord_runs(objects, 4, 1)
    assert removed == 3
    assert chords_of(result) == [
        {0, 1, 2, 3}, {0, 1, 2},
        {0, 1, 2, 3}, {1, 2, 3},
        {0, 1, 2, 3}, {0, 1, 3},
    ]


def test_longer_allowed_runs():
    objects = [s for t in range(0, 900, 150) for s in full_chord(t)]
    result, removed = limit_mania_full_chord_runs(objects, 4, 2)
    assert removed == 2
    assert [len(c) for c in chords_of(result)] == [4, 4, 3, 4, 4, 3]


def test_other_chords_reset_the_run():
    objects = full_chord(0) + [note(0, 150), note(1, 150)] + full_chord(300) + full_chord(450)
    result, removed = limit_mania_full_chord_runs(objects, 4, 1)
    assert removed == 1
    assert [len(c) for c in chords_of(result)] == [4, 2, 4, 3]


def test_hold_notes_are_never_removed():
    objects = full_chord(0) + [hold(3, 150, 600), note(0, 150), note(1, 150), note(2, 150)]
    result, removed = limit_mania_full_chord_runs(objects, 4, 1)
    assert removed == 1
    assert hold(3, 150, 600) in result
    assert note(0, 150) not in result  # the next column in the drop order


def test_seven_keys():
    objects = [s for t in range(0, 600, 150) for s in full_chord(t, 7)]
    result, removed = limit_mania_full_chord_runs(objects, 7, 1)
    assert removed == 2
    assert chords_of(result, 7) == [set(range(7)), set(range(6)), set(range(7)), set(range(1, 7))]
