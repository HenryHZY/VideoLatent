import json

def filter_jsonl(input_file, output_file):
    deleted_count = 0
    filtered_data = []

    # Read the input jsonl file
    with open(input_file, 'r', encoding='utf-8') as infile:
        for line in infile:
            data = json.loads(line)
            # Check the number of 'image_input' entries
            if len(data.get('image_input', [])) in [2, 4]:
                filtered_data.append(data)
            else:
                deleted_count += 1

    # Write the filtered data to the output jsonl file
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for item in filtered_data:
            json.dump(item, outfile, ensure_ascii=False)
            outfile.write("\n")

    return deleted_count

# Example usage
input_file = "../data/data_output3d_begin_10k_resized_clean.jsonl"
output_file = "../data/data_output3d_begin_10k_resized_remove.jsonl"
deleted_count = filter_jsonl(input_file, output_file)

print(f'Deleted {deleted_count} entries where image_input was not 2 or 4.')
