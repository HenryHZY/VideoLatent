import json
import os

def read_jsonl(file_path):
    """Read jsonl file"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def write_jsonl(data, file_path):
    """Write jsonl file"""
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

def process_data(data1_path, data2_path, output_path):
    """Process data matching"""
    # Read data
    data1 = read_jsonl(data1_path)
    data2 = read_jsonl(data2_path)
    
    # Create lookup dictionary for data2, key is (text_input, tuple(image_input))
    data2_dict = {}
    for item in data2:
        key = (item['text_input'], tuple(item['image_input']))
        if key in data2_dict:
            # If key already exists, means data2 has duplicates, store as list
            if not isinstance(data2_dict[key], list):
                data2_dict[key] = [data2_dict[key]]
            data2_dict[key].append(item)
        else:
            data2_dict[key] = item
    
    # Process data1
    processed_data1 = []
    deleted_items = []
    
    for item in data1:
        key = (item['text_input'], tuple(item['image_input']))
        
        if key in data2_dict:
            matched_item = data2_dict[key]
            
            # Check if there are multiple matches
            if isinstance(matched_item, list):
                # Multiple matches, delete this data
                deleted_items.append(item)
                print(f"Deleted data (multiple matches): text_input prefix: {item['text_input'][:50]}...")
            else:
                # Only one match, add idx field
                item['idx'] = matched_item['idx']
                processed_data1.append(item)
        else:
            # No match, keep original data (without adding idx)
            processed_data1.append(item)
    
    # Print statistics
    print(f"Original data1 entries: {len(data1)}")
    print(f"Processed data1 entries: {len(processed_data1)}")
    print(f"Deleted entries: {len(deleted_items)}")
    
    # Save results
    write_jsonl(processed_data1, output_path)
    print(f"Results saved to: {output_path}")
    
    return processed_data1, deleted_items

# Usage example
if __name__ == "__main__":
    # Replace with your actual file paths
    data1_path = "../data/data_output3d_begin_10k_resized_remove.jsonl"  # data1 file path
    data2_path = "../data/idx.jsonl"  # data2 file path
    output_path = "../data/data_output3d_begin_10k_resized.jsonl"  # output file path
    
    # Check if files exist
    if not os.path.exists(data1_path):
        print(f"Error: File {data1_path} does not exist")
        exit(1)
    
    if not os.path.exists(data2_path):
        print(f"Error: File {data2_path} does not exist")
        exit(1)
    
    # Process data
    processed_data, deleted_data = process_data(data1_path, data2_path, output_path)
    
    # Optional: save deleted data for inspection
    if deleted_data:
        deleted_output_path = "deleted_data.jsonl"
        write_jsonl(deleted_data, deleted_output_path)
        print(f"Deleted data saved to: {deleted_output_path}")
