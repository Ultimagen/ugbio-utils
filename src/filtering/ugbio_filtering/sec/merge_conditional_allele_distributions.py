#!/env/python
# Copyright 2022 Ultima Genomics Inc.
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
# DESCRIPTION
#    Merges conditional distributions from multiple files
# CHANGELOG in reverse chronological order

from __future__ import annotations

import argparse
import os.path
import pickle
import sys

from ugbio_filtering.sec.conditional_allele_distribution import ConditionalAlleleDistribution
from ugbio_filtering.sec.conditional_allele_distributions import ConditionalAlleleDistributions
from ugbio_filtering.sec.read_counts import ReadCounts


def get_args(argv: list[str]):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--conditional_allele_distribution_files",
        help="file containing paths to synced conditional allele distribution files (tsv)",
        required=True,
    )
    parser.add_argument(
        "--output_prefix",
        help="prefix to pickle files (per chromosome) " "of serialized ConditionalAlleleDistribution dicts",
        required=True,
    )
    args = parser.parse_args(argv[1:])
    return args


def run(argv: list[str]):
    """
    stack together conditional_allele_distributions from multiple samples into a single file
    """
    args = get_args(argv)

    conditional_allele_distributions = ConditionalAlleleDistributions()
    with open(args.conditional_allele_distribution_files, encoding="utf-8") as cad_files:
        for file_name in cad_files:
            file_name = file_name.strip()  # noqa PLW2901
            if not os.path.exists(file_name):
                continue

            with open(file_name, encoding="utf-8") as file_handle:
                for line in file_handle:
                    (
                        chrom,
                        pos,
                        ground_truth_alleles,
                        true_genotype,
                        observed_alleles,
                        n_samples,
                        allele_counts,
                    ) = line.split("\t")
                    pos = int(pos)
                    allele_counts = allele_counts.split()
                    alleles = allele_counts[0::2]
                    counts = [ReadCounts(*[int(sc) for sc in c.split(",")]) for c in allele_counts[1::2]]
                    allele_counts_dict = dict(zip(alleles, counts, strict=False))
                    conditional_allele_distributions.add_counts(
                        chrom,
                        pos,
                        ConditionalAlleleDistribution(
                            ground_truth_alleles,
                            true_genotype,
                            observed_alleles,
                            allele_counts_dict,
                            int(n_samples),
                        ),
                    )

    with open(f"{args.output_prefix}.txt", "w", encoding="utf-8") as otf:
        for (
            chrom,
            distributions_per_chrom,
        ) in conditional_allele_distributions.distributions_per_chromosome.items():
            for pos, conditional_allele_distribution in distributions_per_chrom.items():
                for record in conditional_allele_distribution.get_string_records(chrom, pos):
                    otf.write(f"{record}\n")
            with open(f"{args.output_prefix}.{chrom}.pkl", "wb") as out_pickle_file:
                pickle.dump(distributions_per_chrom, out_pickle_file)


def main():
    run(sys.argv)


if __name__ == "__main__":
    main()
