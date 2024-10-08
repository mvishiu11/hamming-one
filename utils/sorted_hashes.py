import sys

def process_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    # Skip header line (e.g., "Hashes:")
    data = []
    
    for line in lines[1:]:  # Start from the second line
        parts = line.strip().split(': ')[1].split(' | ')
        true_index = int(parts[0])
        hash_value = int(parts[1])
        
        # Keep only non-negative hashes
        if hash_value >= 0:
            data.append((true_index, hash_value))
    
    # Sort by true_index
    data.sort()
    
    # Output the results
    print(f"Range: {data[0][0]} - {data[-1][0]}")
    for true_index, hash_value in data:
        print(f"{true_index} | {hash_value}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python sorted_hashes.py <file_path>")
    else:
        process_file(sys.argv[1])
