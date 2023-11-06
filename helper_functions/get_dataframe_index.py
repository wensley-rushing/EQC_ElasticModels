# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 15:19:54 2023

@author: Uzo Uwaoma - udu@uw.edu
"""

def find_row(df, target_array):
    # Iterate through the rows of the dataframe
    for index, row in df.iterrows():
        # Check if the values in the row match the target array
        if (row == target_array).all():
            return index  # Return the index if a match is found
