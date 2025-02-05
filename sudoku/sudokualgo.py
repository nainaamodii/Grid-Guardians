"""
This file contains functions to solve the sudoku puzzle
1. valid(puzzle, num, pos) - checks if a number is valid at a given position
    puzzle - 2D array containing rows and columns, 0 for blank space
    num - number to be placed at the given position
    pos - position where the number is to be placed
2. solve_sudoku(puzzle) - solves the sudoku puzzle using backtracking
3. print_board(puzzle) - prints the sudoku puzzle in a formatted way
"""

# function to find if a number is valid at a position in the puzzle
def valid(puzzle, num, pos):
    # check if the number already exists in the row
    for i in range(9):
        if num == puzzle[pos[0]][i] and i != pos[1]:
            return False
    
    # check if the number already exists in the column
    for i in range(9):
        if num == puzzle[i][pos[1]] and i != pos[0]:
            return False
    
    # check if the number already exists in the 3x3 unit
    x = pos[0]//3
    y = pos[1]//3

    for i in range(x*3, x*3 + 3):
        for j in range(y*3, y*3 + 3):
            if num == puzzle[i][j] and (i, j) != pos:
                return False
    
    # if all conditions satisifed then number is valid
    return True

# function to implement backtracking and find solution to the puzzle
def solve_sudoku(puzzle):
    # iterate over the puzzle to find empty cells
    for i in range(9):
        for j in range(9):
            # if the cell is empty then try to fill it with numbers from 1 to 9
            if puzzle[i][j] == 0:
                for num in range(1, 10):
                    # check if the number is valid at the current position
                    if valid(puzzle, num, (i, j)):
                        # if the number is valid then place the number and continue with the next cell
                        puzzle[i][j] = num
                        # recursively solve next cells
                        if solve_sudoku(puzzle):
                            return True
                        # if no number is valid then set current back to 0
                        puzzle[i][j] = 0
                # no number is valid even though empty cells exist
                return False 
    # if all cells are filled then puzzle is solved
    return True

# function to print the solved sudoku puzzle
def print_sudoku(puzzle):
    print('\n')
    for i in range(9):
        for j in range(9):
            print(puzzle[i][j], end = ' ')
            if ((j + 1) % 3 == 0 and j != 8):
                print('|', end = ' ')
        print()
        if ((i + 1) % 3 == 0 and i != 8):
            print('-'*21)