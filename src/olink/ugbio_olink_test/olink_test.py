import argparse
import subprocess
import os
import csv
import boto3
import re
import numpy as np
import pandas as pd
import subprocess
from fuzzysearch import find_near_matches
import multiprocessing as mp



def process_cram_file(cram_file: str, output_path: str, output_name: str="output"):
    """
    Processes a CRAM file and saves the output to the specified path.

    Parameters:
    - cram_file (str): Path to the input CRAM file.
    - output_path (str): Path to save the processed output.

    Returns:
    - None
    """

    sequences = extract_seq(cram_file)

    s3_folder_path = "s3://ultimagen-pipelines-471112545487-us-east-1-runs-data/users-data/gila/to_share/yoav/olink/barcodes_CONFIDENTIAL/"
    local_folder_path = "/data/Runs/olink/"

    copy_cram_files_from_s3(s3_folder_path, local_folder_path)


    barcode_file = '/data/Runs/olink/sample_index_HT_final.v2.csv'
    barcode_mapping = load_barcode_mapping(barcode_file)

    merged_segments, mappings = extract_and_merge_segments(sequences, barcode_mapping, motif_after='TTCCGATCT', motif_before='GCTTGG', segment_length=10)

    fbc_file = '/data/Runs/olink/fbc_HT.csv'
    rbc_file = '/data/Runs/olink/rbc_extended_HT.csv'

    combined_sequences, numerical_values = return_combined_sequences(fbc_file, rbc_file)

    map_identifiers = np.unique(mappings)

    matrix = confusion_matrix_parallel(merged_segments, mappings, map_identifiers, combined_sequences,indels=1)
       
    output_path = os.path.join(output_path, f"{output_name}.npy")

    np.save(output_path, matrix)

    # verify the output exist
    assert os.path.exists(output_path)
    print(f"Output saved to: {output_path}")
    

    # Placeholder for implementation
    pass

def split_workload(n_items, n_splits):
    chunk_size = (n_items + n_splits - 1) // n_splits
    return [(i * chunk_size, min((i + 1) * chunk_size, n_items)) for i in range(n_splits)]

def confusion_matrix_parallel(merged_segments, mappings, map_identifiers, combined_sequences, indels=1):
    counting_matrix = np.zeros((len(map_identifiers), len(combined_sequences)), dtype=int)

    row_ranges = split_workload(len(map_identifiers), mp.cpu_count())
    col_ranges = split_workload(len(combined_sequences), mp.cpu_count())

    args_list = [
        (
            merged_segments,
            mappings,
            map_identifiers[start:end],
            combined_sequences[col_start:col_end],
            indels,
            (start, end),
            (col_start, col_end)
        )
        for start, end in row_ranges
        for col_start, col_end in col_ranges
    ]

    with mp.Pool(processes=mp.cpu_count()) as pool:
        results = pool.map(worker_function, args_list)

    for (row_start, row_end), (col_start, col_end), sub_matrix in results:
        counting_matrix[row_start:row_end, col_start:col_end] = sub_matrix

    return counting_matrix

def worker_function(args):
    (
        merged_segments,
        mappings,
        map_identifiers,
        combined_sequences,
        indels,
        row_range,
        col_range
    ) = args

    sub_matrix = np.zeros((len(map_identifiers), len(combined_sequences)), dtype=int)

    for i, map_identifier in enumerate(map_identifiers):
        seq_i = merged_segments[mappings == map_identifier]
        for j, seq in enumerate(combined_sequences):
            sub_matrix[i, j] = count_indel_matches(seq_i, seq, max_indels=indels)

    return row_range, col_range, sub_matrix



def copy_cram_files_from_s3(s3_folder_path, local_folder_path):
    """
    Copies all files from an S3 folder to a local folder recursively.

    Parameters:
    - s3_folder_path (str): S3 folder path to copy from (e.g., "s3://bucket-name/path/").
    - local_folder_path (str): Local folder path to copy files to.

    Returns:
    - None
    """
    # Ensure local folder exists
    os.makedirs(local_folder_path, exist_ok=True)

    # Initialize S3 client
    s3 = boto3.client('s3')

    # Extract bucket and prefix from the S3 path
    if not s3_folder_path.startswith("s3://"):
        raise ValueError("Invalid S3 folder path. Must start with 's3://'")
    s3_bucket, s3_prefix = s3_folder_path.replace("s3://", "").split("/", 1)

    # List all objects in the S3 folder
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=s3_bucket, Prefix=s3_prefix):
        if "Contents" in page:
            for obj in page["Contents"]:
                s3_key = obj["Key"]
                # Define the local file path
                relative_path = os.path.relpath(s3_key, s3_prefix)
                local_file_path = os.path.join(local_folder_path, relative_path)

                # Ensure local subdirectories exist
                os.makedirs(os.path.dirname(local_file_path), exist_ok=True)

                # Download the file
                s3.download_file(s3_bucket, s3_key, local_file_path)
                print(f"Downloaded: {s3_key} -> {local_file_path}")




def extract_seq(cram_file, max_reads=None, bp_length=None, aws=False):
    env = os.environ.copy()
    if aws:
        # Get AWS credentials
        aws_command = "aws configure export-credentials --format env-no-export"
        aws_process = subprocess.Popen(aws_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        aws_output, aws_error = aws_process.communicate()
        if aws_process.returncode != 0:
            print("Error getting AWS credentials:", aws_error)
            return []
        # Parse the output and update environment variables
        for line in aws_output.strip().split('\n'):
            key, value = line.strip().split('=', 1)
            env[key] = value

    if max_reads is None:
        command = f"samtools view {cram_file}"
    else:
        command = f"samtools view {cram_file} | head -n {max_reads}"


    
    # Rest of your function remains the same
    sequences = []

    # Open the process and read the output
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
    output, error = process.communicate()

    if process.returncode != 0:
        print("Error:", error.decode())
    else:
        # Process the output from samtools
        for line in output.decode().splitlines():
            fields = line.split('\t')
            if len(fields) < 11:
                continue  # Skip lines that don't have enough fields
            seq = fields[9]

            # Extract the first bp_length bases of the read
            if seq:
                if bp_length is None:
                    sequences.append(seq)
                else:
                    sequences.append(seq[:bp_length])

    # print(f"First {bp_length} bp of reads:", sequences[:10])

    return sequences



def load_barcode_mapping(csv_file_path):
    
    # if file is in s3 down load it first
    if csv_file_path.startswith('s3://'):
        s3 = boto3.client('s3')
        bucket, key = csv_file_path.replace("s3://", "").split("/", 1)
        local_file_path = '/tmp/barcode_mapping.csv'
        s3.download_file(bucket, key, local_file_path)
        csv_file_path = local_file_path    
    """
    Loads barcode sequences and their corresponding numerical identifiers from a CSV file.

    Parameters:
    - csv_file_path (str): Path to the CSV file.

    Returns:
    - barcode_dict (dict): Dictionary mapping barcode sequences to numerical identifiers.
    """
    barcode_dict = {}
    with open(csv_file_path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if len(row) >= 2:
                barcode, identifier = row[0].strip(), row[1].strip()
                # Extract numerical part from identifier (e.g., 'i5007' -> 5007)
                if identifier.startswith('i'):
                    numerical_id = int(identifier[1:])
                    barcode_dict[barcode] = numerical_id
    return barcode_dict




def extract_and_merge_segments(sequences, barcode_dict, motif_after='TTCCGATCT', motif_before='GCTTGG', segment_length=10):
    """
    Extracts segments from sequences based on specified motifs, merges them, and maps to barcode identifiers.

    Parameters:
    - sequences (list of str): List of full DNA sequences.
    - barcode_dict (dict): Dictionary mapping barcode sequences to numerical identifiers.
    - motif_after (str): The motif after which to extract segment1.
    - motif_before (str): The motif before which to extract segment2.
    - segment_length (int): The length of segments to extract.

    Returns:
    - merged_segments (list of str): List of merged segments.
    - mappings (list of int): List of numerical identifiers corresponding to each merged segment (-1 if not found).
    """
    merged_segments = []
    mappings = []
    
    # Precompile regex pattern for barcodes for efficient searching
    # Sort barcodes by length in descending order to prioritize longer matches
    sorted_barcodes = sorted(barcode_dict.keys(), key=lambda x: -len(x))
    pattern = re.compile('|'.join(re.escape(barcode) for barcode in sorted_barcodes))
    
    for seq in sequences:
        segment1 = ''
        segment2 = ''
        
        # Extract segment1: after motif_after
        index_after = seq.find(motif_after)
        if index_after != -1:
            start_index = index_after + len(motif_after)
            segment1 = seq[start_index:start_index + segment_length]
        
        # Extract segment2: before motif_before
        index_before = seq.find(motif_before)
        if index_before != -1:
            start_index = max(0, index_before - segment_length)
            segment2 = seq[start_index:index_before]
        
        # Merge segments if both are found
        if segment1 and segment2:
            merged_segment = segment2 + segment1
        else:
            merged_segment = ''
        
        merged_segments.append(merged_segment)
        
        # Initialize mapping as -1 (not found)
        identifier = -1
        
        # Proceed only if segment2 was successfully extracted
        if segment2:
            # Define the search region: after the end of segment2 in the full sequence
            search_region_start = index_before + 10
            search_region = seq[search_region_start:]
            
            # Search for any barcode in the search region
            match = pattern.search(search_region)
            if match:
                barcode = match.group()
                identifier = barcode_dict.get(barcode, -1)
        
        mappings.append(identifier)
    
    is_seq = np.array([bool(seq) for seq in merged_segments])
    is_mapped = np.array([mapping != -1 for mapping in mappings])

    isa = is_seq & is_mapped

    merged_segments = np.array(merged_segments)[isa]
    mappings = np.array(mappings)[isa]

    print("ratio of identified seuqnces: ", sum(isa)/len(isa))

    return merged_segments, mappings

def return_combined_sequences(fbc_file, rbc_file):
    
    # Load the CSV files into pandas DataFrames
    # fb_df = pd.read_csv('/data/Runs/olink/fbc_HT.csv', header=None, names=['FB_seq', 'FB_num'])
    # rb_df = pd.read_csv('/data/Runs/olink/rbc_extended_HT.csv', header=None, names=['RB_seq', 'RB_num'])

    fb_df = pd.read_csv(fbc_file, header=None, names=['FB_seq', 'FB_num'])
    rb_df = pd.read_csv(rbc_file, header=None, names=['RB_seq', 'RB_num'])

    # Convert the numeric part to integer, coerce errors to NaN
    fb_df['FB_num_numeric'] = pd.to_numeric(fb_df['FB_num'].str[2:], errors='coerce')
    rb_df['RB_num_numeric'] = pd.to_numeric(rb_df['RB_num'].str[2:], errors='coerce')

    # Drop rows with NaN values in the numeric columns
    fb_df = fb_df.dropna(subset=['FB_num_numeric'])
    rb_df = rb_df.dropna(subset=['RB_num_numeric'])

    # Convert to integer type after dropping NaNs
    fb_df['FB_num_numeric'] = fb_df['FB_num_numeric'].astype(int)
    rb_df['RB_num_numeric'] = rb_df['RB_num_numeric'].astype(int)

    # Merge on the numeric value column
    merged_df = pd.merge(fb_df, rb_df, left_on='FB_num_numeric', right_on='RB_num_numeric')

    # Create the combined sequence (RB_seq first 8 bp + FB_seq all bp)
    merged_df['Combined_seq'] = merged_df['RB_seq'].str[:8] + merged_df['FB_seq']

    # Extract the lists of combined sequences and numerical integer values
    combined_sequences = merged_df['Combined_seq'].tolist()
    numerical_values = merged_df['FB_num_numeric'].tolist()

    return combined_sequences, numerical_values


def count_indel_matches(seq_i, bc_test, max_indels=1):
    """
    Counts the number of sequences in seq_i that contain bc_test
    or its variants with up to max_indels indels.

    Parameters:
    - seq_i (list of str): List of sequences to search within.
    - bc_test (str): The barcode sequence to search for.
    - max_indels (int): Maximum number of allowed indels.

    Returns:
    - count (int): The total number of matches.
    """
    count = 0
    # Preprocess seq_i if necessary (e.g., build an index)
    for seq in seq_i:
        if approximate_substring_match(seq, bc_test, max_indels):
            count += 1
    return count

def approximate_substring_match(text, pattern, max_indels):
    """
    Checks if the pattern exists within the text allowing up to max_indels indels.

    Parameters:
    - text (str): The sequence to search within.
    - pattern (str): The pattern to search for.
    - max_indels (int): Maximum number of allowed indels.

    Returns:
    - match_found (bool): True if a match is found, False otherwise.
    """
    # Use a suitable approximate matching algorithm
    # For DNA sequences, you can use the fuzzysearch library

    matches = find_near_matches(pattern, text, max_l_dist=max_indels,max_substitutions=0)
    return len(matches) > 0


def main():
    """
    Main function to parse arguments and process the CRAM file.
    """
    parser = argparse.ArgumentParser(description="Process a CRAM file and save the output.")
    parser.add_argument("cram_file", type=str, help="Path to the input CRAM file.")
    parser.add_argument("output_path", type=str, help="Path to save the processed output.")
    parser.add_argument("--output_name", type=str, default="output", help="Name of the output file.")
    
    args = parser.parse_args()
    
    # Call the processing function
    process_cram_file(args.cram_file, args.output_path, args.output_name)

if __name__ == "__main__":
    main()


#   $ python ~/ugbio-utils/src/olink_test/ugbio_olink_test/onlink_test.py /data/Runs/olink/409959-Olink_HT31-Z0008.1M.bam /data/Runs/olink/ --output_name mat1
#   $ uv run python ~/ugbio-utils/src/olink_test/ugbio_olink_test/onlink_test.py /data/Runs/olink/409959-Olink_HT31-Z0008.1M.bam /data/Runs/olink/ --output_name mat1


# docker run -v /data:/data  olink_test_docker olink_test /data/Runs/olink/409959-Olink_HT31-Z0008.1M.bam /data/Runs/olink/ --output_name mat1

# after change code:
# cd ~/ugbio-utils/
# docker build -t olink_docker -f src/olink/Dockerfile .
# docker run -v /data:/data  olink_docker olink_test /data/Runs/olink/409959-Olink_HT31-Z0008.1M.bam /data/Runs/olink/ --output_name mat1