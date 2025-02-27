# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
import json

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fields", type=str, nargs='+', default=[], required=True)
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    args = parser.parse_args()
    return args

def main():
    args = get_args()
    print (f"{args}")

    field2int = { field:i for i, field in enumerate(args.fields) }
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.input_file, 'r') as f, open(args.output_file, 'w') as outfile:
        for line in f:
            prediction = json.loads(line.strip())
            predict_int = field2int[prediction["predict"]] if prediction["predict"] in field2int else field2int[args.fields[0]]
            outfile.write(str(predict_int) + "\n")
    
if __name__ == "__main__":
    main()