#!/usr/bin/env python3

import argparse

def parse_matrix_file(filename):
    """Parse the matrix file and return a list of columns"""
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    # Transpose the matrix to get columns instead of rows
    matrix = []
    for line in lines:
        # Remove whitespace and split by spaces
        row = line.strip().split()
        # Convert to integers
        row = [int(x) for x in row]
        matrix.append(row)
    
    # Transpose to get columns
    if matrix:
        num_cols = len(matrix[0])
        columns = []
        for col_idx in range(num_cols):
            column = [row[col_idx] for row in matrix]
            columns.append(column)
        return columns
    else:
        return []

def convert_1d_to_2d(index_1d):
    """Convert 1D index (0-15) to 2D index (row, col) for 4x4 matrix"""
    row = index_1d // 4
    col = index_1d % 4
    return row, col

def generate_instructions(columns):
    """Generate bufferBlockStrassen instructions from the matrix columns"""
    instructions = []
    
    for col_idx, column in enumerate(columns):
        # Find non-zero elements and their indices
        non_zero_indices = []
        non_zero_signs = []
        
        for idx, value in enumerate(column):
            if value != 0:
                non_zero_indices.append(idx)
                non_zero_signs.append(1 if value > 0 else 0)
        
        # Determine which bufferBlockStrassen function to use based on number of non-zero elements
        num_non_zero = len(non_zero_indices)
        
        if num_non_zero == 0:
            # Skip columns with all zeros
            continue
        elif num_non_zero == 1:
            # bufferBlockStrassen_1: single element
            row1, col1 = convert_1d_to_2d(non_zero_indices[0])
            instruction = f"bufferBlockStrassen_1(stream_M, buffer_c[{row1}][{col1}], {non_zero_signs[0]});"
        elif num_non_zero == 2:
            # bufferBlockStrassen_2: two elements
            row1, col1 = convert_1d_to_2d(non_zero_indices[0])
            row2, col2 = convert_1d_to_2d(non_zero_indices[1])
            instruction = f"bufferBlockStrassen_2(stream_M, buffer_c[{row1}][{col1}], {non_zero_signs[0]}, buffer_c[{row2}][{col2}], {non_zero_signs[1]});"
        elif num_non_zero == 3:
            # bufferBlockStrassen_3: three elements
            row1, col1 = convert_1d_to_2d(non_zero_indices[0])
            row2, col2 = convert_1d_to_2d(non_zero_indices[1])
            row3, col3 = convert_1d_to_2d(non_zero_indices[2])
            instruction = f"bufferBlockStrassen_3(stream_M, buffer_c[{row1}][{col1}], {non_zero_signs[0]}, buffer_c[{row2}][{col2}], {non_zero_signs[1]}, buffer_c[{row3}][{col3}], {non_zero_signs[2]});"
        elif num_non_zero == 4:
            # bufferBlockStrassen_4: four elements
            row1, col1 = convert_1d_to_2d(non_zero_indices[0])
            row2, col2 = convert_1d_to_2d(non_zero_indices[1])
            row3, col3 = convert_1d_to_2d(non_zero_indices[2])
            row4, col4 = convert_1d_to_2d(non_zero_indices[3])
            instruction = f"bufferBlockStrassen_4(stream_M, buffer_c[{row1}][{col1}], {non_zero_signs[0]}, buffer_c[{row2}][{col2}], {non_zero_signs[1]}, buffer_c[{row3}][{col3}], {non_zero_signs[2]}, buffer_c[{row4}][{col4}], {non_zero_signs[3]});"
        else:
            # For more than 4 non-zero elements, we'll need to handle differently
            # For now, let's use bufferBlockStrassen_4 with the first 4 elements
            row1, col1 = convert_1d_to_2d(non_zero_indices[0])
            row2, col2 = convert_1d_to_2d(non_zero_indices[1])
            row3, col3 = convert_1d_to_2d(non_zero_indices[2])
            row4, col4 = convert_1d_to_2d(non_zero_indices[3])
            instruction = f"bufferBlockStrassen_4(stream_M, buffer_c[{row1}][{col1}], {non_zero_signs[0]}, buffer_c[{row2}][{col2}], {non_zero_signs[1]}, buffer_c[{row3}][{col3}], {non_zero_signs[2]}, buffer_c[{row4}][{col4}], {non_zero_signs[3]});"
            print(f"Warning: Column {col_idx} has {num_non_zero} non-zero elements, using first 4")
        
        instructions.append(instruction)
    
    return instructions

def main():
    parser = argparse.ArgumentParser(description='Generate bufferBlockStrassen instructions from W4 matrix')
    parser.add_argument('--input', type=str, default='W4_matrix.txt', help='Input matrix file')
    parser.add_argument('--output', type=str, default='W4_instructions.txt', help='Output instructions file')
    
    args = parser.parse_args()
    
    print(f"Reading matrix from {args.input}...")
    columns = parse_matrix_file(args.input)
    
    print(f"Generating instructions...")
    instructions = generate_instructions(columns)
    
    print(f"Writing {len(instructions)} instructions to {args.output}...")
    with open(args.output, 'w') as f:
        for instruction in instructions:
            f.write(instruction + '\n')
    
    print("Done!")

if __name__ == "__main__":
    main()
