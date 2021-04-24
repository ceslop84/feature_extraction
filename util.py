import os
import cv2


SIZE = 224

def create_dir(dir):
    if not os.path.isdir(dir):
            try:
                os.mkdir(dir)
            except:
                print(f"Creation of directory {dir} failed!")
                exit(1)

def create_dir(dir):
    if not os.path.isdir(dir):
            try:
                os.mkdir(dir)
            except:
                print(f"Creation of directory {dir} failed!")
                exit(1)

def save(data_dict, path_dict, output_folder):
    for cat,img_list in data_dict.items():
        for i, data in enumerate(img_list):
            path = path_dict.get(cat)[i]
            folder_tree = path.split(".")[0]
            file_name = folder_tree.split("/")[-1]
            output_file = f"{output_folder}/{file_name}.txt"
            np.savetxt(output_file, data)

def crop_resize(img_file):
        img = cv2.imread(img_file, cv2.IMREAD_COLOR)
        y, x, c = img.shape
        if y < x:
            aux = int((x-y)/2)
            img = img[:,aux:aux+y]
        elif x < y:
            aux = int((y-x)/2)
            img = img[aux:aux+x,:]
        return cv2.resize(img, (SIZE, SIZE))

def load_images_from_file(file):
    # takes all images and convert them to grayscale.
    # return a dictionary that holds all images category by category.
    images = {}
    files = {}
    with open(file) as f:
        for line in f:
            img_path = line.split()[0]
            cat = line.split()[1]
            img = crop_resize(img_path)
            if img is not None:
                cat_img_list = images.get(cat)
                cat_fl_list = files.get(cat)
                if (cat_img_list is not None) and (cat_fl_list is not None):
                    cat_img_list.append(img)
                    cat_fl_list.append(img_path)
                else:
                    cat_img_list = []
                    cat_fl_list = []
                    cat_img_list.append(img)
                    cat_fl_list.append(img_path)
                    images[cat] = cat_img_list
                    files[cat] = cat_fl_list
    return images, files
