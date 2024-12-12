import argparse
import sys


def run(argv):  # noqa PLR0915
    """
    Generate a config file for controlFREEC given a set of parameters
    """

    args = parse_args(argv)

    config_text = ""
    ### general section ###
    config_text = "\n".join([config_text, "", "[general]"])
    s = "BedGraphOutput=" + args.BedGraphOutput
    config_text = "\n".join([config_text, s])
    if args.NaiveNormalization:
        s = "NaiveNormalization=" + args.NaiveNormalization
        config_text = "\n".join([config_text, s])
    s = "chrLenFile=" + args.chrLenFile
    config_text = "\n".join([config_text, s])
    s = "contaminationAdjustment=" + args.contaminationAdjustment
    config_text = "\n".join([config_text, s])
    if args.contamination:
        s = "contamination=" + args.contamination
        config_text = "\n".join([config_text, s])
    s = "maxThreads=" + args.maxThreads
    config_text = "\n".join([config_text, s])
    s = "window=" + args.window
    config_text = "\n".join([config_text, s])
    s = "chrFiles=" + args.chrFiles
    config_text = "\n".join([config_text, s])
    s = "degree=" + args.degree
    config_text = "\n".join([config_text, s])
    if args.forceGCcontentNormalization:
        s = "forceGCcontentNormalization=" + args.forceGCcontentNormalization
        config_text = "\n".join([config_text, s])
    if args.sex:
        s = "sex=" + args.sex
        config_text = "\n".join([config_text, s])
    if args.ploidy:
        s = "ploidy=" + args.ploidy
        config_text = "\n".join([config_text, s])
    if args.gemMappabilityFile:
        s = "gemMappabilityFile=" + args.gemMappabilityFile
        config_text = "\n".join([config_text, s])

    ### sample section ###
    config_text = "\n".join([config_text, "", "[sample]"])
    s = "mateFile=" + args.sample_mateFile
    config_text = "\n".join([config_text, s])
    s = "mateCopyNumberFile=" + args.sample_mateCopyNumberFile
    config_text = "\n".join([config_text, s])
    s = "miniPileup=" + args.sample_miniPileupFile
    config_text = "\n".join([config_text, s])
    s = "inputFormat=" + args.sample_inputFormat
    config_text = "\n".join([config_text, s])
    s = "mateOrientation=" + args.sample_mateOrientation
    config_text = "\n".join([config_text, s])

    ### control section ###
    config_text = "\n".join([config_text, "", "[control]"])
    s = "mateFile=" + args.control_mateFile
    config_text = "\n".join([config_text, s])
    s = "mateCopyNumberFile=" + args.control_mateCopyNumberFile
    config_text = "\n".join([config_text, s])
    s = "miniPileup=" + args.control_miniPileupFile
    config_text = "\n".join([config_text, s])
    s = "inputFormat=" + args.control_inputFormat
    config_text = "\n".join([config_text, s])
    s = "mateOrientation=" + args.control_mateOrientation
    config_text = "\n".join([config_text, s])

    ### BAF section ###
    config_text = "\n".join([config_text, "", "[BAF]"])
    # s = 'makePileup=' + args.baf_makePileup
    # config_text = '\n'.join([config_text, s])
    s = "fastaFile=" + args.baf_fastaFile
    config_text = "\n".join([config_text, s])
    s = "SNPfile=" + args.baf_SNPfile
    config_text = "\n".join([config_text, s])
    if args.baf_makePileup:
        s = "makePileup=" + args.baf_makePileup
        config_text = "\n".join([config_text, s])

    with open(args.sample_name + ".config.txt", "w") as text_file:
        text_file.write(config_text)


def parse_args(argv):
    parser = argparse.ArgumentParser(
        prog="generate_controlFREEC_config",
        description="Generate a config file for controlFREEC given a set of parameters",
    )
    parser.add_argument("--sample_name", help="sample_name", required=True)
    parser.add_argument("--BedGraphOutput", help="general:BedGraphOutput", required=True)
    parser.add_argument("--NaiveNormalization", help="general:NaiveNormalization", required=False)
    parser.add_argument("--chrLenFile", help="general:chrLenFile", required=True)
    parser.add_argument("--contaminationAdjustment", help="general:contaminationAdjustment", required=True)
    parser.add_argument("--contamination", help="general:contamination", required=False)
    parser.add_argument("--maxThreads", help="general:maxThreads", required=True)
    parser.add_argument("--window", help="general:window", required=True)
    parser.add_argument("--chrFiles", help="general:chrFiles", required=True)
    parser.add_argument("--degree", help="general:degree", required=True)
    parser.add_argument("--forceGCcontentNormalization", help="general:forceGCcontentNormalization", required=False)
    parser.add_argument("--sex", help="general:sex", required=False)
    parser.add_argument("--ploidy", help="general:ploidy", required=False)
    parser.add_argument("--gemMappabilityFile", help="general:gemMappabilityFile", required=False)
    parser.add_argument("--sample_mateFile", help="sample:mateFile", required=True)
    parser.add_argument("--sample_mateCopyNumberFile", help="sample:mateCopyNumberFile", required=True)
    parser.add_argument("--sample_miniPileupFile", help="sample:miniPileup", required=True)
    parser.add_argument("--sample_inputFormat", help="sample:inputFormat", required=True)
    parser.add_argument("--sample_mateOrientation", help="sample:mateOrientation", required=True)
    parser.add_argument("--control_mateFile", help="control:mateFile", required=True)
    parser.add_argument("--control_mateCopyNumberFile", help="control:mateCopyNumberFile", required=True)
    parser.add_argument("--control_miniPileupFile", help="control:miniPileup", required=True)
    parser.add_argument("--control_inputFormat", help="control:inputFormat", required=True)
    parser.add_argument("--control_mateOrientation", help="control:mateOrientation", required=True)
    parser.add_argument("--baf_makePileup", help="baf:makePileup", required=False)
    parser.add_argument("--baf_fastaFile", help="baf:fastaFile", required=True)
    parser.add_argument("--baf_SNPfile", help="baf:SNPfile", required=True)
    args = parser.parse_args(argv[1:])
    return args


def main():
    run(sys.argv)


if __name__ == "__main__":
    run(sys.argv)
