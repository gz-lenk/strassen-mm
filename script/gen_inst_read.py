#!/usr/bin/env python3

import argparse

def parse_matrix_file(filename):
    """Parse the matrix file and return a list of rows"""
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    matrix = []
    for line in lines:
        # Remove whitespace and split by spaces
        row = line.strip().split()
        # Convert to integers
        row = [int(x) for x in row]
        matrix.append(row)
    
    return matrix

def generate_instructions(matrix):
    """Generate addTile instructions from the matrix"""
    instructions = []
    
    for row_idx, row in enumerate(matrix):
        # Find non-zero elements and their indices
        non_zero_indices = []
        non_zero_values = []
        
        for idx, value in enumerate(row):
            if value != 0:
                non_zero_indices.append(idx)
                non_zero_values.append(value)
        
        # Determine which addTile function to use based on number of non-zero elements
        num_non_zero = len(non_zero_indices)
        
        if num_non_zero == 0:
            # Skip rows with all zeros
            continue
        elif num_non_zero == 1:
            # addTile_1: single element
            sign = 1 if non_zero_values[0] > 0 else 0
            instruction = f"addTile_1(input_B, {non_zero_indices[0]}, {sign}, stream_B);"
        elif num_non_zero == 2:
            # addTile_2: two elements
            sign1 = 1 if non_zero_values[0] > 0 else 0
            sign2 = 1 if non_zero_values[1] > 0 else 0
            instruction = f"addTile_2(input_B, {non_zero_indices[0]}, {non_zero_indices[1]}, {sign1}, {sign2}, stream_B);"
        elif num_non_zero == 3:
            # addTile_3: three elements
            sign1 = 1 if non_zero_values[0] > 0 else 0
            sign2 = 1 if non_zero_values[1] > 0 else 0
            sign3 = 1 if non_zero_values[2] > 0 else 0
            instruction = f"addTile_3(input_B, {non_zero_indices[0]}, {non_zero_indices[1]}, {non_zero_indices[2]}, {sign1}, {sign2}, {sign3}, stream_B);"
        elif num_non_zero == 4:
            # addTile_4: four elements
            sign1 = 1 if non_zero_values[0] > 0 else 0
            sign2 = 1 if non_zero_values[1] > 0 else 0
            sign3 = 1 if non_zero_values[2] > 0 else 0
            sign4 = 1 if non_zero_values[3] > 0 else 0
            instruction = f"addTile_4(input_B, {non_zero_indices[0]}, {non_zero_indices[1]}, {non_zero_indices[2]}, {non_zero_indices[3]}, {sign1}, {sign2}, {sign3}, {sign4}, stream_B);"
        else:
            # For more than 4 non-zero elements, we'll need to handle differently
            # For now, let's use addTile_4 with the first 4 elements
            sign1 = 1 if non_zero_values[0] > 0 else 0
            sign2 = 1 if non_zero_values[1] > 0 else 0
            sign3 = 1 if non_zero_values[2] > 0 else 0
            sign4 = 1 if non_zero_values[3] > 0 else 0
            instruction = f"addTile_4(input_B, {non_zero_indices[0]}, {non_zero_indices[1]}, {non_zero_indices[2]}, {non_zero_indices[3]}, {sign1}, {sign2}, {sign3}, {sign4}, stream_B);"
            print(f"Warning: Row {row_idx} has {num_non_zero} non-zero elements, using first 4")
        
        instructions.append(instruction)
    
    return instructions

def main():
    # argparse.add_argument("--input", type=str, default="U4_matrix.txt")

    input_file = "V4_matrix.txt"
    output_file = "V4_instructions.txt"
    
    print(f"Reading matrix from {input_file}...")
    matrix = parse_matrix_file(input_file)
    
    print(f"Generating instructions...")
    instructions = generate_instructions(matrix)
    
    print(f"Writing {len(instructions)} instructions to {output_file}...")
    with open(output_file, 'w') as f:
        for instruction in instructions:
            f.write(instruction + '\n')
    
    print("Done!")

if __name__ == "__main__":
    main()
