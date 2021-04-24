import os
import subprocess
from util import create_dir


TEST = False

def mpeg7(input_file, output_folder, test=False):

    globals_list = globals()
    globals_list['TEST'] = test

    # Running the tool with the defaults values.
    desc_list = ["CSD", "SCD", "CLD", "DCD", "HTD", "EHD"]

    # Output data.
    create_dir(output_folder)
    mpeg_folder = f"{output_folder}/mpeg/"
    create_dir(mpeg_folder)
    file_tmp = f"{mpeg_folder}/input_file.txt"

    #Removing the class (0, 1 and 2) info from the image list and creating a temp file.
    f_list = list()
    with open(input_file, "r") as f:
        for i, line in enumerate(f):
            f_list.append(line.split(" ")[0])
            if i>10:
                break
    with open(file_tmp, "w") as f:
        for obj in f_list:
            f.write(f"{obj}\n")

    if os.name == "posix":
            subprocess.run("./env.sh", shell=True)

    for d in desc_list:
        if os.name == "nt":
            subprocess.run(f"mpeg7/windows/MPEG7Fex.exe {d} {file_tmp} {mpeg_folder}/{d}.txt", shell=True)
        elif os.name == "posix":
            subprocess.run(f"./mpeg7/linux/MPEG7Fex {d} {file_tmp} {mpeg_folder}/{d}.txt", shell=True)
        else:
            print("Sorry, OS not supported!")
            exit(1)

    # Removing the temp file.
    os.remove(file_tmp)

if __name__ == '__main__':
    mpeg7('Images/labels_small.txt', "Output", True)


