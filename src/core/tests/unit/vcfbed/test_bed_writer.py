from pathlib import Path

from ugbio_core.vcfbed.bed_writer import BedWriter


class TestBedWriter:
    def test_write(self, tmpdir):
        file_name = Path(tmpdir) / "example.bed"
        writer = BedWriter(file_name)
        writer.write("chr2", 100, 102, "hmer-indel")
        writer.write("chr3", 120, 123, "snp")
        writer.close()

        lines = open(file_name).readlines()
        assert "chr2\t100\t102\thmer-indel\n" == lines[0]
        assert "chr3\t120\t123\tsnp\n" == lines[1]
