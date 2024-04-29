#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 17:02:32 2024

@author: emmanuel
"""

def decode_text(filename):
    """
    Reads all lines from a file and returns the decoded message using the key: last element of triangulated order of integers.
    
    Parameters:
        filename (str): The path to the file to be read.
    
    Returns:
        a str ofthe decoded message from the file.
    """
    
    decode_dict = {}
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            lines = file.readlines()
        # Strip newline characters from each line
        for line in lines:
            num, word = line.strip().split()
            decode_dict[int(num)] = word

        n = len(decode_dict)
        """
        Generates a list of last elements from each row in a triangular arrangement of numbers 1 to n.
        Returns:
            list: A list containing the last number of each row.
        """
        last_elements = []
        current_number = 1
        row_number = 1
        
        while current_number <= n:
            next_row_end = current_number + row_number - 1  # Calculate the last number of the current row
            if next_row_end > n:  # If the calculated end exceeds n, adjust it to n
                next_row_end = n
            last_elements.append(next_row_end)
            current_number = next_row_end + 1  # Start the next row after the last number of the current row
            row_number += 1  # Each new row has one more element than the last

        text = ''
        for decoder_key in last_elements:
            if len(text) != 0:
                text = text +" " + decode_dict[decoder_key]
            else:
                text = text + decode_dict[decoder_key]
        return text
    except FileNotFoundError:
        print(f"Error: The file '{filename}' does not exist.")
        return
    except Exception as e:
        print(f"An error occurred: {e}")
        return



# Example usage:
# Assuming you have a file named 'example.txt' in the same directory as your script
if __name__ == "__main__":
    file_path = './coding_qual_input.txt'
    file_lines = decode_text(file_path)
    print(file_lines)
