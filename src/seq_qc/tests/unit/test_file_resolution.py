import pytest
from ugbio_seq_qc.file_resolution import (
    _split_s3_uri,
    find_sample_basename,
    resolve_sample_files,
)


class TestSplitS3Uri:
    def test_bucket_and_prefix(self):
        assert _split_s3_uri("s3://my-bucket/path/to/dir/") == ("my-bucket", "path/to/dir/")

    def test_no_trailing_slash(self):
        assert _split_s3_uri("s3://my-bucket/path/to/dir") == ("my-bucket", "path/to/dir/")

    def test_bucket_root(self):
        assert _split_s3_uri("s3://my-bucket") == ("my-bucket", "")


class TestFindSampleBasename:
    def test_prefers_single_cram(self):
        files = ["sampleA.cram", "sampleA.cram.crai", "sampleA.json", "sampleA.csv"]
        assert find_sample_basename(files) == "sampleA"

    def test_excludes_unmatched_cram(self):
        files = ["sampleA.cram", "X_unmatched.cram", "sampleA.json", "sampleA.csv"]
        assert find_sample_basename(files) == "sampleA"

    def test_falls_back_to_json_csv_pair(self):
        files = ["sampleA.json", "sampleA.csv", "sampleA.applicationQC.json"]
        assert find_sample_basename(files) == "sampleA"

    def test_ignores_applicationqc_json(self):
        # only the applicationQC json has no plain-csv partner -> still resolves the real pair
        files = ["sampleA.json", "sampleA.csv", "applicationQC.json", "applicationQC.csv"]
        assert find_sample_basename(files) == "sampleA"

    def test_raises_when_ambiguous(self):
        files = ["a.json", "a.csv", "b.json", "b.csv"]
        with pytest.raises(ValueError, match="Multiple JSON"):
            find_sample_basename(files)

    def test_raises_when_none(self):
        with pytest.raises(ValueError, match="Could not determine"):
            find_sample_basename(["readme.txt"])


class TestResolveSampleFilesLocal:
    def test_resolves_local_dir(self, tmp_path):
        (tmp_path / "sampleA.json").write_text("{}")
        (tmp_path / "sampleA.csv").write_text("Mean_cvg,30\n")
        json_path, csv_path, basename = resolve_sample_files(tmp_path)
        assert basename == "sampleA"
        assert json_path == tmp_path / "sampleA.json"
        assert csv_path == tmp_path / "sampleA.csv"

    def test_raises_when_missing(self, tmp_path):
        (tmp_path / "sampleA.json").write_text("{}")
        with pytest.raises(ValueError):
            resolve_sample_files(tmp_path)
