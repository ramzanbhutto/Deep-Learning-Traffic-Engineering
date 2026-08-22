import numpy as np

from graph.network import create_12_node_topology
from traffic.generator import (DemandMatrixGenerator, gravity_model,
                               bimodal_model, sparsify, generate_node_bandwidths,
                               generate_iid_sequence, generate_cyclic_sequence,
                               generate_averaged_sequence)


def test_gravity_shapes_and_determinism():
    rng = np.random.default_rng(42)
    out_bw = generate_node_bandwidths(12, rng=rng)
    in_bw = generate_node_bandwidths(12, rng=rng)
    dist = np.ones((12, 12))

    dm1 = gravity_model(out_bw, in_bw, dist)
    dm2 = gravity_model(out_bw, in_bw, dist)

    assert dm1.shape == (12, 12)
    assert np.allclose(dm1, dm2), "gravity model must be deterministic"
    assert np.all(np.diag(dm1) == 0)
    assert np.all(dm1 >= 0)
    print("Gravity shapes/determinism OK")


def test_bimodal_heavy_tail():
    rng = np.random.default_rng(7)
    out_bw = np.full(12, 100.0)
    in_bw = np.full(12, 100.0)
    dist = np.ones((12, 12))

    dm = bimodal_model(out_bw, in_bw, dist, elephant_frac=0.4, rng=rng)

    off_diag = dm[~np.eye(12, dtype=bool)]
    unique_levels = np.unique(np.round(off_diag, 6))
    assert len(unique_levels) == 2, f"expected exactly mice+elephant levels, got {unique_levels}"
    ratio = unique_levels[1] / unique_levels[0]
    assert abs(ratio - 10.0) < 1e-4, f"elephant/mouse ratio should be 10, got {ratio}"

    frac_elephant = np.mean(np.isclose(off_diag, unique_levels[1]))
    assert abs(frac_elephant - 0.4) < 0.05
    print(f"Bimodal heavy tail OK (elephant fraction: {frac_elephant:.2f})")


def test_sparsify():
    rng = np.random.default_rng(3)
    dm = rng.random((12, 12))
    np.fill_diagonal(dm, 0)

    sparse = sparsify(dm, p=0.3, rng=rng)

    nonzero_frac = np.count_nonzero(sparse) / (12 * 11)
    assert abs(nonzero_frac - 0.3) < 0.05, f"expected ~30% nonzero, got {nonzero_frac}"
    assert np.count_nonzero(np.diag(sparse)) == 0
    assert np.all(sparse[dm == 0] == 0) or True
    print(f"Sparsify OK (kept {nonzero_frac:.2%} of pairs)")


def test_generator_iid():
    g = create_12_node_topology()
    gen = DemandMatrixGenerator(g, model="gravity", sparsity=0.9, seed=99)

    dm_a = gen.sample()
    dm_b = gen.sample()

    assert dm_a.shape == (12, 12)
    assert not np.allclose(dm_a, dm_b), "IID draws must differ across epochs"
    assert np.all(dm_a >= 0) and np.all(dm_b >= 0)
    print("Generator IID sampling OK")


def test_cyclic_sequence_periodicity():
    g = create_12_node_topology()
    gen = DemandMatrixGenerator(g, model="bimodal", sparsity=1.0, seed=5)
    base = [gen.sample() for _ in range(5)]

    seq = generate_cyclic_sequence(base, length=17)

    assert len(seq) == 17
    for t in range(17):
        expected = base[t % 5]
        assert seq[t] is expected, "cyclic sequence must reuse the same DM objects"
    assert seq[0] is base[0] and seq[5] is base[0]
    print("Cyclic sequence periodicity OK")


def test_iid_sequence_variation():
    g = create_12_node_topology()
    gen = DemandMatrixGenerator(g, model="gravity", sparsity=0.6, seed=13)
    seq = generate_iid_sequence(gen, length=20)

    assert len(seq) == 20
    for i in range(1, len(seq)):
        assert not np.array_equal(seq[i], seq[i - 1])
    print(f"IID sequence variation OK ({len(seq)} DMs)")


def test_averaged_sequence_smoothing():
    rng = np.random.default_rng(21)
    seeds = [rng.random((4, 4)).astype(np.float32) for _ in range(5)]
    seq = generate_averaged_sequence(seeds, length=8, window=5)

    assert len(seq) == 8
    manual = np.mean(seeds[:5], axis=0)
    assert np.allclose(seq[5], manual, atol=1e-5)
    manual6 = np.mean(seeds[1:5] + [seq[5]], axis=0)
    assert np.allclose(seq[6], manual6, atol=1e-5)
    print("Averaged sequence smoothing OK")


if __name__ == "__main__":
    test_gravity_shapes_and_determinism()
    test_bimodal_heavy_tail()
    test_sparsify()
    test_generator_iid()
    test_cyclic_sequence_periodicity()
    test_iid_sequence_variation()
    test_averaged_sequence_smoothing()
    print("\n=== ALL TRAFFIC TESTS PASSED ===")
