import os

def add_data_to_line(file_name, line_number, data):
    try:
        with open(file_name, 'r+') as file:
            lines = file.readlines()
            if line_number <= len(lines):
                lines[line_number-1] = lines[line_number-1].rstrip('\n') + data + '\n'
                file.seek(0)
                file.writelines(lines)
                file.truncate()
                print("The data has been successfully added to the line.")
            else:
                print("Invalid line number.")
    except IOError:
        print("An error occurred while writing to the file.")


def add_lines_to_file(file_name, lines):
    try:
        with open(file_name, 'a') as file:
            for line in lines:
                file.write(line + '\n')
        print("The lines have been successfully added to the file.")
    except IOError:
        print("An error occurred while writing to the file.")

# Example of using the function
#file_name = "holy_code/test.txt"
#lines = ["This is the first line.", "This is the second line."]

#add_lines_to_file(file_name, lines)

#data = "BOMBOCLAAT"

#add_data_to_line(file_name, 3, data)