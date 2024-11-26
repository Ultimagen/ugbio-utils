import itertools
import os
from enum import Enum
from os.path import basename, dirname, splitext
from os.path import join as pjoin

import numpy as np
import pandas as pd
import pyfaidx
import pysam
from simppl.simple_pipeline import SimplePipeline
from ugbio_core.consts import (
    ALT,
    CHROM,
    DEFAULT_FLOW_ORDER,
    FILTER,
    IS_CYCLE_SKIP,
    POS,
    QUAL,
    REF,
    FileExtension,
)
from ugbio_core.dna_sequence_utils import get_max_softclip_len
from ugbio_core.exec_utils import print_and_execute
from ugbio_core.logger import logger
from ugbio_core.sorter_utils import read_effective_coverage_from_sorter_json
from ugbio_core.vcfbed.variant_annotation import (
    VcfAnnotator,
    get_cycle_skip_dataframe,
    get_motif_around_snv,
)
from ugbio_ppmseq.ppmSeq_utils import PpmseqStrandVcfAnnotator


class FeatureMapFields(Enum):
    CHROM = CHROM
    POS = POS
    REF = REF
    ALT = ALT
    QUAL = QUAL
    FILTER = FILTER
    READ_COUNT = "X_READ_COUNT"
    FILTERED_COUNT = "X_FILTERED_COUNT"
    X_SCORE = "X_SCORE"
    X_EDIST = "X_EDIST"
    X_LENGTH = "X_LENGTH"
    X_MAPQ = "X_MAPQ"
    X_INDEX = "X_INDEX"
    X_FC1 = "X_FC1"
    X_FC2 = "X_FC2"
    X_QUAL = "X_QUAL"
    X_RN = "X_RN"
    TRINUC_CONTEXT_WITH_ALT = "trinuc_context_with_alt"
    HMER_CONTEXT_REF = "hmer_context_ref"
    HMER_CONTEXT_ALT = "hmer_context_alt"
    IS_CYCLE_SKIP = IS_CYCLE_SKIP
    IS_FORWARD = "is_forward"
    IS_DUPLICATE = "is_duplicate"
    MAX_SOFTCLIP_LENGTH = "max_softclip_length"
    X_FLAGS = "X_FLAGS"
    X_CIGAR = "X_CIGAR"
    PREV_1 = "prev_1"
    PREV_2 = "prev_2"
    PREV_3 = "prev_3"
    NEXT_1 = "next_1"
    NEXT_2 = "next_2"
    NEXT_3 = "next_3"


class FeatureMapFilters(Enum):
    LOW_QUAL = "LowQual"
    SINGLE_READ = "SingleRead"
    PASS = "PASS"  # noqa: S105
    PRE_FILTERED = "PreFiltered"


def get_hmer_of_central_base(sequence: str) -> int:
    """
    Get the length of the homopolymer spanning the central base in the given sequence.
    Examples:
    ACA -> 1
    ACC -> 2
    ACCCT -> 3
    AAAAAT -> 5

    Parameters
    ----------
    sequence : str
        The sequence to check. Must be of odd length, so the central base is well defined.

    Returns
    -------
    int
        The length of the homopolymer run of the central base.
    """
    if not isinstance(sequence, str):
        raise TypeError("sequence must be a string")
    if len(sequence) % 2 != 1:
        raise ValueError("The sequence length must be odd.")
    if len(sequence) < 1:
        raise ValueError("The sequence length must be at least 1.")
    hmer_lengths = np.array([sum(1 for _ in x[1]) for x in itertools.groupby(sequence)])
    central_hmer = hmer_lengths[np.argmax(np.cumsum(hmer_lengths) > len(sequence) // 2)]
    return int(central_hmer)


def is_biallelic_snv(record: pysam.VariantRecord) -> bool:
    """
    Check if a given VCF record is a biallelic SNV.

    Parameters
    ----------
    record : pysam.VariantRecord
        The VCF record to check.

    Returns
    -------
    bool
        True if the record is a biallelic SNV, False otherwise.
    """
    if not isinstance(record, pysam.VariantRecord):
        raise TypeError("record must be a pysam.VariantRecord")
    return len(record.ref) == 1 and len(record.alts) == 1 and len(record.alts[0]) == 1


class FeaturemapAnnotator(VcfAnnotator):
    """
    Annotate vcf with featuremap-specific fields derived from X_FLAGS and X_CIGAR:
    - is_forward: is the read forward mapped
    - is_duplicate: is the read a duplicate
    - max_softclip_length: maximum softclip length
    """

    def __init__(self):
        pass

    def edit_vcf_header(self, header: pysam.VariantHeader) -> pysam.VariantHeader:
        """
        Edit the VCF header to include new fields

        Parameters
        ----------
        header : pysam.VariantHeader
            VCF header

        Returns
        -------
        pysam.VariantHeader
            Modified VCF header

        """
        header.add_line("##ugbio_mrd.mrd_utils.py_FeaturemapAnnotator=.")
        header.add_line(
            f"##INFO=<ID={FeatureMapFields.IS_FORWARD.value},"
            'Number=0,Type=Flag,Description="is the read forward mapped">'
        )
        header.add_line(
            f"##INFO=<ID={FeatureMapFields.IS_DUPLICATE.value},"
            'Number=0,Type=Flag,Description="is the read a duplicate">'
        )
        header.add_line(
            f"##INFO=<ID={FeatureMapFields.MAX_SOFTCLIP_LENGTH.value},"
            'Number=1,Type=Integer,Description="maximum softclip length between start and end of read">'
        )

        return header

    def process_records(self, records: list[pysam.VariantRecord]) -> list[pysam.VariantRecord]:
        """

        Parameters
        ----------
        records : list[pysam.VariantRecord]
            list of VCF records

        Returns
        -------
        list[pysam.VariantRecord]
            list of updated VCF records
        """
        records_out = [None] * len(records)
        for j, record in enumerate(records):
            if FeatureMapFields.X_FLAGS.value in record.info:
                flags = record.info[FeatureMapFields.X_FLAGS.value]
                record.info[FeatureMapFields.IS_FORWARD.value] = (flags & 16) == 0
                record.info[FeatureMapFields.IS_DUPLICATE.value] = (flags & 1024) != 0
            if FeatureMapFields.X_CIGAR.value in record.info:
                record.info[FeatureMapFields.MAX_SOFTCLIP_LENGTH.value] = get_max_softclip_len(
                    record.info[FeatureMapFields.X_CIGAR.value]
                )
            records_out[j] = record

        return records_out


class RefContextVcfAnnotator(VcfAnnotator):
    def __init__(
        self,
        ref_fasta: str,
        flow_order: str = DEFAULT_FLOW_ORDER,
        motif_length_to_annotate: int = 3,
        max_hmer_length: int = 20,
    ):
        """
        Annotator to add reference context to VCF records, only modifies biallelic SNVs.
        The following are added to the INFO field:
        - trinuc_context_with_alt: reference trinucleotide context
        - prev_N: base i in the reference before the variant, i in range 1 to N=length motif_length_to_annotate
        - next_N: base i in the reference after the variant, i in range 1 to N=length motif_length_to_annotate
        - hmer_context_ref: reference homopolymer context, up to length max_hmer_length
        - hmer_context_alt: homopolymer context in the ref allele (assuming the variant considered only),
            up to length max_hmer_length
        - is_cycle_skip: True if the SNV is a cycle skip

        Parameters
        ----------
        ref_fasta : str
            Path to the reference FASTA file.
        flow_order : str
            Flow order of the flow cell.
        motif_length_to_annotate : int, optional
            The length of the motif to annotate context up to (prev_N / next_N). Defaults to 3.
        max_hmer_length : int, optional
            The maximum length of the homopolymer to annotate context up to. Defaults to 20.
        """
        # check inputs
        if len(flow_order) != 4:  # noqa: PLR2004
            raise ValueError(f"Flow order must be of length 4, got {flow_order}")
        if not os.path.isfile(ref_fasta):
            raise FileNotFoundError(f"Reference FASTA file not found: {ref_fasta}")

        # save inputs
        self.ref_fasta = ref_fasta
        self.motif_length_to_annotate = motif_length_to_annotate
        self.max_hmer_length = max_hmer_length
        self.flow_order = flow_order

        # init accesory objects
        self.cycle_skip_dataframe = get_cycle_skip_dataframe(flow_order)

        # info field names
        self.TRINUC_CONTEXT_WITH_ALT = FeatureMapFields.TRINUC_CONTEXT_WITH_ALT.value
        self.HMER_CONTEXT_REF = FeatureMapFields.HMER_CONTEXT_REF.value
        self.HMER_CONTEXT_ALT = FeatureMapFields.HMER_CONTEXT_ALT.value
        self.CYCLE_SKIP_FLAG = FeatureMapFields.IS_CYCLE_SKIP.value
        self.SINGLE_REF_BASES = []
        for i in range(self.motif_length_to_annotate):
            self.SINGLE_REF_BASES.append(f"prev_{i + 1}")
            self.SINGLE_REF_BASES.append(f"next_{i + 1}")

        self.info_fields_to_add = [
            self.TRINUC_CONTEXT_WITH_ALT,
            self.HMER_CONTEXT_REF,
            self.HMER_CONTEXT_ALT,
        ] + self.SINGLE_REF_BASES
        # self.CYCLE_SKIP_FLAG not included because it's a flag, only there if True

    def edit_vcf_header(self, header: pysam.VariantHeader) -> pysam.VariantHeader:
        """
        Edit the VCF header to include new fields

        Parameters
        ----------
        header : pysam.VariantHeader
            VCF header

        Returns
        -------
        pysam.VariantHeader
            Modified VCF header

        """
        header.add_line(
            "##ugbio_core.vcfbed.variant_annotation._RefContextVcfAnnotator="
            f"ref:{os.path.basename(self.ref_fasta)}"
            f"_motif_length_to_annotate:{self.motif_length_to_annotate}"
            f"_max_hmer_length:{self.max_hmer_length}"
        )

        header.add_meta(
            "INFO",
            items=[
                ("ID", self.TRINUC_CONTEXT_WITH_ALT),
                ("Number", 1),
                ("Type", "String"),
                ("Description", "reference trinucleotide context and alt base"),
            ],
        )

        for i in range(self.motif_length_to_annotate):
            header.add_meta(
                "INFO",
                items=[
                    ("ID", f"prev_{i + 1}"),
                    ("Number", 1),
                    ("Type", "String"),
                    ("Description", f"{i + 1} bases in the reference before variant"),
                ],
            )
            header.add_meta(
                "INFO",
                items=[
                    ("ID", f"next_{i + 1}"),
                    ("Number", 1),
                    ("Type", "String"),
                    ("Description", f"{i + 1} bases in the reference after variant"),
                ],
            )

        header.add_meta(
            "INFO",
            items=[
                ("ID", self.HMER_CONTEXT_REF),
                ("Number", 1),
                ("Type", "Integer"),
                ("Description", f"reference homopolymer context, up to length {self.max_hmer_length}"),
            ],
        )

        header.add_meta(
            "INFO",
            items=[
                ("ID", self.HMER_CONTEXT_ALT),
                ("Number", 1),
                ("Type", "Integer"),
                (
                    "Description",
                    f"homopolymer context in the ref allele (assuming the variant considered only), "
                    f"up to length {self.max_hmer_length}",
                ),
            ],
        )

        header.add_meta(
            "INFO",
            items=[
                ("ID", self.CYCLE_SKIP_FLAG),
                ("Number", 0),
                ("Type", "Flag"),
                ("Description", "True if the SNV is a cycle skip"),
            ],
        )

        return header

    def process_records(self, records: list[pysam.VariantRecord]) -> list[pysam.VariantRecord]:
        """
        Parameters
        ----------
        records : list[pysam.VariantRecord]
            list of VCF records

        Returns
        -------
        list[pysam.VariantRecord]
            list of updated VCF records
        """
        records_out = [None] * len(records)
        faidx_ref = pyfaidx.Fasta(self.ref_fasta, build_index=False, rebuild=False)
        for j, record in enumerate(records):
            if is_biallelic_snv(record):
                # get motif
                ref_around_snv = get_motif_around_snv(record, self.max_hmer_length, faidx_ref)
                central_base_ind = len(ref_around_snv) // 2
                trinuc_ref = ref_around_snv[central_base_ind - 1 : central_base_ind + 2]
                # create sequence of alt allele
                alt_around_snv_list = list(ref_around_snv)
                alt_around_snv_list[central_base_ind] = record.alts[0]
                alt_around_snv = "".join(alt_around_snv_list)
                trinuc_alt = alt_around_snv[central_base_ind - 1 : central_base_ind + 2]
                # assign to record
                record.info[self.HMER_CONTEXT_REF] = get_hmer_of_central_base(ref_around_snv)
                record.info[self.HMER_CONTEXT_ALT] = get_hmer_of_central_base(alt_around_snv)
                record.info[self.TRINUC_CONTEXT_WITH_ALT] = trinuc_ref + record.alts[0]

                for index in range(1, self.motif_length_to_annotate + 1):
                    field_name = f"next_{index}"
                    record.info[field_name] = ref_around_snv[central_base_ind + index]
                    field_name = f"prev_{index}"
                    record.info[field_name] = ref_around_snv[central_base_ind - index]

                is_cycle_skip = self.cycle_skip_dataframe.at[(trinuc_ref, trinuc_alt), IS_CYCLE_SKIP]  # noqa: PD008
                record.info[self.CYCLE_SKIP_FLAG] = is_cycle_skip

                # make sure all the info fields are present
                for info_field in self.info_fields_to_add:
                    if info_field not in record.info:
                        raise ValueError(
                            f"INFO field {info_field} was supposed to be added to VCF record but was not found"
                        )
            records_out[j] = record

        return records_out


def create_hom_snv_featuremap(
    featuremap: str,
    sorter_stats_json: str = None,
    hom_snv_featuremap: str = None,
    sp: SimplePipeline = None,
    requested_min_coverage: int = 20,
    min_af: float = 0.7,
):
    """Create a HOM SNV featuremap from a featuremap

    Parameters
    ----------
    featuremap : str
        Input featuremap.
    sorter_stats_json : str
        Path to Sorter statistics JSON file, used to extract the median coverage. If None (default), minimum coverage
        will be set to requested_min_coverage even if the median coverage is lower, might yield an empty output.
    hom_snv_featuremap : str, optional
        Output featuremap with HOM SNVs reads to be used as True Positives. If None (default),
        the hom_snv_featuremap will be the same as the input featuremap with a ".hom_snv.vcf.gz" suffix.
    sp : SimplePipeline, optional
        SimplePipeline object to use for running commands. If None (default), commands will be run using subprocess.
    requested_min_coverage : int, optional
        Minimum coverage requested for locus to be propagated to the output. If the median coverage is lower than this
        value, the median coverage will be used as the minimum coverage instead.
        Default 20
    min_af : float, optional
        Minimum allele fraction in the featuremap to be considered a HOM SNV
        Default 0.7
        The default is chosen as 0.7 and not higher because some SNVs are pre-filtered from the FeatureMap due to
        MAPQ<60 or due to adjacent hmers.
    """

    # check inputs
    if not os.path.isfile(featuremap):
        raise FileNotFoundError(f"featuremap {featuremap} does not exist")
    if sorter_stats_json:
        if not os.path.isfile(sorter_stats_json):
            raise FileNotFoundError(f"sorter_stats_json {sorter_stats_json} does not exist")
    if hom_snv_featuremap is None:
        if featuremap.endswith(".vcf.gz"):
            hom_snv_featuremap = featuremap[: -len(".vcf.gz")]
        hom_snv_featuremap = featuremap + ".hom_snv.vcf.gz"
    hom_snv_bed = hom_snv_featuremap.replace(".vcf.gz", ".bed.gz")
    logger.info(f"Writting HOM SNV featuremap to {hom_snv_featuremap}")

    # get minimum coverage
    if sorter_stats_json:
        (
            _,
            _,
            _,
            min_coverage,
            _,
        ) = read_effective_coverage_from_sorter_json(sorter_stats_json, min_coverage_for_fp=requested_min_coverage)
    else:
        min_coverage = requested_min_coverage
    logger.info(
        f"Using a minimum coverage of {min_coverage} for HOM SNV featuremap (requested {requested_min_coverage})"
    )

    # Create commands to filter the featuremap for homozygous SNVs.
    cmd_get_hom_snv_loci_bed_file = (
        # Use bcftools to query specific fields in the vcf file. This includes the chromosome (CHROM),
        # the 0-based start position (POS0), the 1-based start position (POS), and the number of reads
        # in the locus (X_READ_COUNT) for the specified feature map.
        f"bcftools query -f '%CHROM\t%POS0\t%POS\t%INFO/{FeatureMapFields.READ_COUNT.value}\n' {featuremap} |"
        # Pipe the output to bedtools groupby command.
        # Here, -c 3 means we are specifying the third column as the key to groupby.
        # The '-full' option includes all columns from the input in the output.
        # The '-o count' option is specifying to count the number of lines for each group.
        f"bedtools groupby -c 3 -full -o count | "
        # Pipe the result to an awk command, which filters the result based on minimum coverage and allele frequency.
        # The '$4>=~{min_coverage}' part checks if the fourth column (which should be read count) is greater than or
        # equal to the minimum coverage. The '$5/$4>=~{min_af}' part checks if the allele frequency (calculated as
        # column 5 divided by column 4) is greater than or equal to the minimum allele frequency.
        f"awk '($4>={min_coverage})&&($5/$4>={min_af})' | "
        # The final output is then compressed and saved to the specified location in .bed.gz format.
        f"gzip > {hom_snv_bed}"
    )
    cmd_intersect_bed_file_with_original_featuremap = (
        f"bedtools intersect -a {featuremap} -b {hom_snv_bed} -u -header | bcftools view - -Oz -o {hom_snv_featuremap}"
    )
    cmd_index_hom_snv_featuremap = f"bcftools index -ft {hom_snv_featuremap}"

    # Run the commands
    try:
        for command in (
            cmd_get_hom_snv_loci_bed_file,
            cmd_intersect_bed_file_with_original_featuremap,
            cmd_index_hom_snv_featuremap,
        ):
            print_and_execute(command, simple_pipeline=sp, module_name=__name__)

    finally:
        # remove temp file
        if os.path.isfile(hom_snv_bed):
            os.remove(hom_snv_bed)


def filter_featuremap_with_bcftools_view(
    input_featuremap_vcf: str,
    intersect_featuremap_vcf: str,
    min_coverage: int = None,
    max_coverage: int = None,
    regions_file: str = None,
    bcftools_include_filter: str = None,
    sp: SimplePipeline = None,
) -> str:
    """
    Create a bcftools view command to filter a featuremap vcf

    Parameters
    ----------
    input_featuremap_vcf : str
        Path to input featuremap vcf
    intersect_featuremap_vcf : str
        Path to output intersected featuremap vcf
    min_coverage : int, optional
        Minimum coverage to include, by default None
    max_coverage : int, optional
        Maximum coverage to include, by default None
    regions_file : str, optional
        Path to regions file, by default None
    bcftools_include_filter: str, optional
        bcftools include filter to apply as part of a "bcftools view <vcf> -i 'pre_filter_bcftools_include'"
        before sampling, by default None
    sp : SimplePipeline, optional
        SimplePipeline object to use for printing and running commands, by default None
    """
    bcftools_view_command = f"bcftools view {input_featuremap_vcf} -O z -o {intersect_featuremap_vcf} "
    include_filters = [bcftools_include_filter.replace("'", '"')] if bcftools_include_filter else []
    # filter out variants with coverage outside of min and max coverage
    if min_coverage is not None:
        include_filters.append(f"(INFO/{FeatureMapFields.READ_COUNT.value} >= {min_coverage})")
    if max_coverage is not None:
        include_filters.append(f"(INFO/{FeatureMapFields.READ_COUNT.value} <= {max_coverage})")

    bcftools_include_string = " && ".join(include_filters)
    if bcftools_include_string:
        bcftools_view_command += f" -i '{bcftools_include_string}' "
    if regions_file:
        bcftools_view_command += f" -T {regions_file} "
    bcftools_index_command = f"bcftools index -t {intersect_featuremap_vcf}"

    print_and_execute(bcftools_view_command, simple_pipeline=sp, module_name=__name__, shell=True)  # noqa: S604
    print_and_execute(
        bcftools_index_command,
        simple_pipeline=sp,
        module_name=__name__,
    )
    if not os.path.isfile(intersect_featuremap_vcf):
        raise FileNotFoundError(f"failed to create {intersect_featuremap_vcf}")


def featuremap_to_dataframe(  # noqa: C901, PLR0912 #TODO: refactor
    featuremap_vcf: str,
    output_file: str = None,
    input_info_fields: list[str] = "all",
    input_format_fields: list[str] = None,
    sample_name: str = None,
    sample_index: int = None,
    default_int_fillna_value: int = 0,
    default_string_fillna_value: int = np.nan,
):
    """
    Converts featuremap in vcf format to dataframe

    Parameters
    ----------
    featuremap_vcf : str
        Path to featuremap vcf file
    output_file : str, optional
        Path to output file, by default None
        If None, saved to the same basename as featuremap_vcf with .parquet extension
    input_info_fields : list[str], optional
        List of input info fields to include in dataframe, by default 'all''
        If 'all' then all the info fields are read to columns
        If None then no info fields are read to columns
    input_format_fields : list[str], optional
        List of format info fields to include in dataframe, by default None
        If 'all' then all the info fields are read to columns
        If None then no info fields are read to columns
    sample_name : str, optional
        Name of sample to read formats columns from, default None
        If sample_name and sample_index are both None then no formats columns are read
    sample_index : int, optional
        Index of sample to read formats columns from, default None
        If sample_name and sample_index are both None and format_info_fields is not empty
        then sample_index defaults to 0
    default_int_fillna_value : int, optional
        Value to fill na values with for Integer fields, by default 0
    default_string_fillna_value : str, optional
        Value to fill na values with for String fields, by default np.nan

    Returns
    -------
    pd.DataFrame
        Dataframe of featuremap

    Raises
    ------
    ValueError
        If sample_name and sample_index are both not None
        If sample_name is not in vcf file
        If sample_index is out of range
        If input_info_fields contains a field not in vcf header
        If format_info_fields contains a field not in vcf header

    """
    # check inputs
    if sample_name is not None and sample_index is not None:
        raise ValueError("Cannot specify both sample_name and sample_index")
    if sample_name is not None and sample_name not in pysam.VariantFile(featuremap_vcf).header.samples:
        raise ValueError(f"Sample {sample_name} not in vcf file")
    if sample_index is not None and sample_index >= len(pysam.VariantFile(featuremap_vcf).header.samples):
        raise ValueError(f"Sample index {sample_index} out of range")
    if input_info_fields is not None and input_info_fields != "all" and len(input_info_fields) > 0:
        for info_field in input_info_fields:
            if info_field not in pysam.VariantFile(featuremap_vcf).header.info:
                raise ValueError(f"Input info field {info_field} not in vcf header")
    if input_format_fields is not None and input_format_fields != "all" and len(input_format_fields) > 0:
        for info_field in input_format_fields:
            if info_field not in pysam.VariantFile(featuremap_vcf).header.formats:
                raise ValueError(f"Format info field {info_field} not in vcf header")
        if sample_index is None and sample_name is None:
            sample_index = 0
        if sample_index is None and sample_name is not None:
            sample_index = list(pysam.VariantFile(featuremap_vcf).header.samples).index(sample_name)

    # define type conversion dictionary, String is converted to object to support np.nan
    type_conversion_dict = {
        "String": object,
        "Integer": int,
        "Float": float,
        "Flag": bool,
    }

    # auxiliary function to parse info and formats fields to read
    def get_metadata_dict(values: list[str], input_fields: list[str] = None):
        if input_fields is None:
            return {}, []
        metadata_dict_full = {value.name: (value.number, value.type) for value in values}
        input_fields_to_use = list(metadata_dict_full.keys()) if input_fields.lower() == "all" else input_fields
        for key in input_fields_to_use:
            if key not in metadata_dict_full:
                raise ValueError(f"Input field {key} missing from vcf header")
        metadata_dict = {key: metadata_dict_full[key] for key in input_fields_to_use}
        return metadata_dict, input_fields_to_use

    # read vcf file to dataframe
    with pysam.VariantFile(featuremap_vcf) as vcf_handle:
        # parse header and get metadata
        header = vcf_handle.header
        info_metadata_dict, info_input_fields_to_use = get_metadata_dict(
            values=header.info.values(), input_fields=input_info_fields
        )
        format_metadata_dict, format_input_fields_to_use = get_metadata_dict(
            values=header.formats.values(), input_fields=input_format_fields
        )
        # TODO remove this patch once the issue is fixed in FeatureMap
        # This section is a patch for specific integer tags, since there is an issue with FeautreMap
        # not propagating the types of copied SAM tags properly, so that they all default to String.
        # There is a pending request to fix, in the mean time this is a workaround.
        harcoded_info_fields = {x: "Integer" for x in ("a3", "ae", "as", "s2", "s3", "te", "ts")}
        harcoded_info_fields["rq"] = "Float"
        for field, datatype in harcoded_info_fields.items():
            if field in info_metadata_dict:
                info_metadata_dict[field] = (info_metadata_dict[field][0], datatype)
        # read vcf file to dataframe
        featuremap_df = pd.DataFrame(
            vcf_row_generator(
                vcf_handle=vcf_handle,
                info_metadata_dict=info_metadata_dict,
                formats_metadata_dict=format_metadata_dict,
                sample_index=sample_index,
            ),
            columns=[CHROM, POS, REF, ALT, QUAL, FILTER] + info_input_fields_to_use + format_input_fields_to_use,
        )
    # fillna in int columns before converting types, because int columns can't contain NaNs
    featuremap_df = featuremap_df.fillna(
        {
            info_key: default_int_fillna_value
            for info_key in info_metadata_dict.keys()
            if type_conversion_dict[info_metadata_dict[info_key][1]] is int
        }
    )
    featuremap_df.fillna(
        {
            info_key: default_string_fillna_value
            for info_key in info_metadata_dict.keys()
            if type_conversion_dict[info_metadata_dict[info_key][1]] is object
        }
    )
    # convert types
    featuremap_df = featuremap_df.astype(
        {
            "qual": float,
            **{
                info_key: type_conversion_dict[info_metadata_dict[info_key][1]]
                for info_key in info_metadata_dict.keys()
                if info_metadata_dict[info_key][0]
            },
        }
    )
    # Determine output file name
    if output_file is None:
        if featuremap_vcf.endswith(FileExtension.VCF_GZ.value):
            output_file = pjoin(
                dirname(featuremap_vcf),
                splitext(splitext(basename(featuremap_vcf))[0])[0] + FileExtension.PARQUET.value,
            )
        else:
            output_file = pjoin(
                dirname(featuremap_vcf),
                splitext(basename(featuremap_vcf))[0] + FileExtension.PARQUET.value,
            )
    elif not output_file.endswith(FileExtension.PARQUET.value):
        output_file = f"{output_file}{FileExtension.PARQUET.value}"

    # Save and return
    featuremap_df.to_parquet(output_file)
    return featuremap_df


def annotate_featuremap(
    input_featuremap: str,
    output_featuremap: str,
    ref_fasta: str,
    motif_length_to_annotate: int,
    max_hmer_length: int,
    flow_order: str = DEFAULT_FLOW_ORDER,
    ppmseq_adapter_version: str = None,
    process_number: int = 0,
):
    """
    Annotate featuremap with ref context, hmer length and ppmSeq features

    Parameters
    ----------
    input_featuremap: str
        Input featuremap file
    output_featuremap: str
        Output featuremap file
    ref_fasta: str
        Reference fasta file
    motif_length_to_annotate: int
        Motif length to annotate
    max_hmer_length: int
        Max hmer length
    flow_order: str, optional
        Flow order, default TGCA
    ppmSeq_adapter_version: str, optional
        ppmSeq adapter version, if None no ppmSeq annotation is performed
    """
    featuremap_annotator = FeaturemapAnnotator()
    ref_context_annotator = RefContextVcfAnnotator(
        ref_fasta=ref_fasta,
        flow_order=flow_order,
        motif_length_to_annotate=motif_length_to_annotate,
        max_hmer_length=max_hmer_length,
    )
    annotators = [featuremap_annotator, ref_context_annotator]
    if ppmseq_adapter_version:
        ppmseq_annotator = PpmseqStrandVcfAnnotator(adapter_version=ppmseq_adapter_version)
        annotators.append(ppmseq_annotator)
    VcfAnnotator.process_vcf(
        annotators=annotators,
        input_path=input_featuremap,
        output_path=output_featuremap,
        process_number=process_number,
    )


def vcf_row_generator(
    vcf_handle: pysam.VariantFile,
    info_metadata_dict: dict = None,
    formats_metadata_dict: dict = None,
    sample_index: int = None,
):
    """
    Return a generator object that reads entries from an open vcf file handle and yields a list of values

    Parameters
    ----------
    vcf_handle: pysam.VariantFile
        An open vcf file handle
    info_metadata_dict: dict
        A dictionary of info columns to read from the vcf file. The keys are the info column names and the values
        default None, in which case no INFO fields are read
    formats_metadata_dict: dict
        A dictionary of format columns to read from the vcf file. The keys are the info column names and the values
        default None, in which case no FORMAT fields are read
    sample_index: int
        The index of the sample to read from the vcf file.

    """
    for record in vcf_handle.fetch():
        # read basic fields
        entry = [
            record.chrom,
            record.pos,
            record.ref,
            record.alts[0] if len(record.alts) >= 1 else None,
            record.qual,
            ";".join(record.filter.keys()),
        ]
        # read info fields
        if info_metadata_dict is not None and len(info_metadata_dict) > 0:
            entry += [
                record.info.get(info_key)[0]
                if info_value[0] == "A"
                and record.info.get(info_key, None) is not None
                and len(record.info.get(info_key)) >= 1
                else record.info.get(info_key, np.nan)
                for info_key, info_value in info_metadata_dict.items()
            ]
        # read format fields
        if formats_metadata_dict is not None and len(formats_metadata_dict) > 0:
            entry += [
                record.samples[sample_index].get(formats_key, np.nan) for formats_key in formats_metadata_dict.keys()
            ]
        yield entry
