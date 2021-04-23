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
img_list_tmp = "image_list.txt"
output_folder = "Output"

# Running the tool with the defaults values.
param_list = ["CSD", "SCD", "CLD", "DCD", "HTD", "EHD"]

# Output data.
create_dir(f"Output/")
create_dir(f"Output/{name}/")

#Removing the class (0, 1 and 2) info from the image list and creating a temp file.
f_list = list()
with open(img_list, "r") as f:
    for line in f:
        f_list.append(line.split(" ")[0])
with open(img_list_tmp, "w") as f:
    for obj in f_list:
        f.write(f"{obj}\n")

for
    if os.name == "nt":
        subprocess.run(f"mpeg7/windows/MPEG7Fex.exe CSD 64 {img_list_out} CSD_64.txt", shell=True)
    elif os.name == "posix":
        subprocess.run("sh env.sh", shell=True)
        subprocess.run(f"./mpeg7/linux/MPEG7Fex CSD 64 {img_list_out} CSD_64.txt", shell=True)
    else:
        print("Sorry, OS not supported!")
        exit(1)

# Removing the temp file.
os.remove(img_list_out)





