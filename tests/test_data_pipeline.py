from prefix_ttt.data_pipeline import micro_batches


def test_micro_batches_partition_one_group_without_repeats():
    order = list(range(133))
    for cursor in (0, 128):
        for world_size in (1, 4, 8):
            batches = [batch for rank in range(world_size)
                       for batch in micro_batches(order, cursor, rank, world_size, 8)]
            flat = sum(batches, [])
            assert sorted(flat) == order[cursor:cursor + 128]
            assert len(flat) == len(set(flat))
            assert all(len(batch) <= 8 for batch in batches)


def test_micro_batches_keep_manifest_order():
    order = list(range(256))
    assert micro_batches(order, 128, 0, 2, 4)[0] == [128, 130, 132, 134]
    assert micro_batches(order, 128, 1, 2, 4)[0] == [129, 131, 133, 135]
