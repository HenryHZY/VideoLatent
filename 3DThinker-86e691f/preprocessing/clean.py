import json
import re

def filter_jsonl_file(input_file, output_file):
    """
    Read JSONL file and filter out invalid data rows
    
    Args:
        input_file: Input JSONL file path
        output_file: Output JSONL file path
    """
    valid_lines = []
    removed_count = 0
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                # Parse JSON data
                data = json.loads(line.strip())
                
                # Check if text_output field exists
                if 'text_output' not in data:
                    print(f"Warning: Line {line_num} missing text_output field")
                    removed_count += 1
                    continue
                
                text_output = data['text_output']
                
                # Count occurrences of <output_3D>
                output_3d_count = text_output.count('<output_3D>')
                
                # Check if contains </output_3D>
                contains_closing_tag = '</output_3D>' in text_output
                
                # Determine if this data should be removed
                # Remove condition: <output_3D> count != 1 or contains </output_3D>
                if output_3d_count != 1 or contains_closing_tag:
                    print(f"Remove line {line_num}: <output_3D> count={output_3d_count}, contains </output_3D>={contains_closing_tag}")
                    removed_count += 1
                else:
                    # Keep this data
                    valid_lines.append(line.strip())
                    
            except json.JSONDecodeError as e:
                print(f"Error: Line {line_num} JSON parsing failed: {e}")
                removed_count += 1
                continue
    
    # Write valid data to new file
    with open(output_file, 'w', encoding='utf-8') as f:
        for line in valid_lines:
            f.write(line + '\n')
    
    print(f"Processing completed!")
    print(f"Original data rows: {len(valid_lines) + removed_count}")
    print(f"Retained data rows: {len(valid_lines)}")
    print(f"Removed data rows: {removed_count}")

# Usage example
if __name__ == "__main__":
    input_file = "../data/data_output3d_begin_10k_resized_raw.jsonl"  # Replace with your input file path
    output_file = "../data/data_output3d_begin_10k_resized_clean.jsonl"  # Replace with your desired output file path
    
    filter_jsonl_file(input_file, output_file)
