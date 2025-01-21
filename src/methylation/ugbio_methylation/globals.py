from dataclasses import dataclass, fields
from pathlib import Path

import pandas as pd

H5_FILE = ".methyl_seq.applicationQC.h5"
HTML_REPORT = ".methyl_seq.applicationQC.html"
BASE_PATH = Path(__file__).parent  # should be: src/methylation/ugbio_methylation
TEMPLATE_NOTEBOOK = BASE_PATH / "reports" / "methyldackel_qc_report.ipynb"


@dataclass
class MethylDackelConcatenationCsvs:
    mbias: str
    mbias_non_cpg: str
    merge_context: str
    merge_context_non_cpg: str
    per_read: str | None = None

    def iterate_fields(self):
        for field in fields(self):
            field_name = field.name
            field_value = getattr(self, field_name)
            yield field_name, field_value

    def get_keys_to_convert(self) -> pd.Series:
        suffix = "_desc"
        fields = self.__dataclass_fields__
        if self.per_read is None:
            fields = [field for field in fields if field != "per_read"]
        field_names_with_suffix = [f"{field}{suffix}" for field in fields]
        return pd.Series(field_names_with_suffix)
