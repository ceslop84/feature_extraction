import os
import subprocess

def create_dir(dir):
    if not os.path.isdir(dir):
            try:
                os.mkdir(dir)
            except:
                print(f"Creation of directory {dir} failed!")
                exit(1)

# Parameters.
img_list = "Images/labels_small.txt"
output_folder = "Output"

# Running the tool with the defaults values.
desc_list = ["CSD", "SCD", "CLD", "DCD", "HTD", "EHD"]

# Output data.
create_dir(output_folder)
mpeg_folder = f"{output_folder}/mpeg/"
create_dir(mpeg_folder)
file_tmp = f"{mpeg_folder}/img_list.txt"

#Removing the class (0, 1 and 2) info from the image list and creating a temp file.
f_list = list()
with open(img_list, "r") as f:
    for line in f:
        f_list.append(line.split(" ")[0])
with open(file_tmp, "w") as f:
    for obj in f_list:
        f.write(f"{obj}\n")

for d in desc_list:
    if os.name == "nt":
        subprocess.run(f"mpeg7/windows/MPEG7Fex.exe {d} {file_tmp} {mpeg_folder}/{d}.txt", shell=True)
    elif os.name == "posix":
        subprocess.run("sh env.sh", shell=True)
        subprocess.run(f"./mpeg7/linux/MPEG7Fex CSD 64 {d} {file_tmp} {mpeg_folder}/{d}.txt", shell=True)
    else:
        print("Sorry, OS not supported!")
        exit(1)

# Removing the temp file.
os.remove(img_list_out)





