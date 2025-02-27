import ugbio_core.vcfbed.genotype as genotype


def test_different_gt():
    assert not genotype.different_gt("0|1", (0, 1))
    assert not genotype.different_gt("1|0", (0, 1))
    assert genotype.different_gt("1|1", (0, 1))
    assert genotype.different_gt("1/2", (0, 1))
