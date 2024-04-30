#!/usr/bin/env python3

import json
import sys

def convert_json(omics_json, output_json):
    with open(omics_json, 'r') as file:
        omics_dict = json.load(file)

    output_dict = {}
    for item in omics_dict:
        output_dict[item['key']] = item['value']

    # Write the simplified dictionary to a new JSON file
    with open(output_json, 'w') as file:
        json.dump(output_dict, file, indent=2)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python omics_param_json_to_standard_json.py input_json output_json")
        sys.exit(1)
    
    input_json = sys.argv[1]
    output_json = sys.argv[2]

    convert_json(input_json, output_json)
