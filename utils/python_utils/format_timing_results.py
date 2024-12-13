import os
import json
import argparse

def parse_experiment_lines(file_path, output_directory):
    """
    Parses lines in the file that start with '###' and extracts key-value pairs to save in a JSON file.

    Args:
        file_path (str): Path to the input file.
        output_directory (str): Directory where the JSON file will be saved.

    Returns:
        str: Path to the saved JSON file.

    Raises:
        FileNotFoundError: If the input file does not exist.
        ValueError: If the output directory does not exist.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Input file not found: {file_path}")

    if not os.path.isdir(output_directory):
        raise ValueError(f"Output directory does not exist: {output_directory}")

    # Prepare to store parsed results
    experiments = []

    # Read and process the file
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if line.startswith('###'):
                try:
                    # Remove '###' and split into key-value pairs
                    content = line[3:]
                    pairs = content.split(',')
                    parsed_entry = {}
                    for pair in pairs:
                        key, value = pair.split(':')
                        key = key.strip()
                        value = value.strip()
                        # If the key is 'testcase' and value contains '/', extract the last level directory
                        if key == 'testCase' and '/' in value:
                            value = value.split('/')[-1]
                        parsed_entry[key] = value
                    experiments.append(parsed_entry)

                    '''
                    # Remove '###' and split into key-value pairs
                    content = line[3:]
                    pairs = content.split(',')
                    parsed_entry = {pair.split(':')[0].strip(): pair.split(':')[1].strip() for pair in pairs}
                    experiments.append(parsed_entry)
                    '''
                except (IndexError, ValueError):
                    print(f"Skipping malformed line: {line}")

    # Prepare the output file path
    output_file = os.path.join(output_directory, 'ell_experiments.json')

    # Save parsed results to a JSON file
    with open(output_file, 'w') as json_file:
        json.dump(experiments, json_file, indent=4)

    return output_file

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse lines starting with '###' in a file and save as JSON.")
    parser.add_argument("file_path", type=str, help="Path to the input file.")
    parser.add_argument("output_directory", type=str, help="Directory where the JSON file will be saved.")
    
    args = parser.parse_args()

    try:
        output_file = parse_experiment_lines(args.file_path, args.output_directory)
        print(f"Parsed data saved to {output_file}")
    except Exception as e:
        print(f"Error: {e}")

