import os
import numpy as np
import cv2
from utils_gist import *
from extract_patches.core import extract_patches

TEST = True

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

def gist_img(img_dict):
    # Creates descriptors using sift library
    # Takes one parameter that is images dictionary
    # Return an array whose first index holds the decriptor_list without an order
    # And the second index holds the sift_vectors dictionary which holds the descriptors but this is seperated class by class
    gist_vectors = dict()
    descriptor_list = list()
    gist = GistUtils()
    for key,value in img_dict.items():
        features = list()
        for i, img in enumerate(value):
            des = gist.get_gist_vec(img)
            if des is not None:
                descriptor_list.extend(des)
                features.append(des)
            if TEST and i>10:
                break
        gist_vectors[key] = np.float32(features)

    return ["gist", np.float32(descriptor_list), gist_vectors]

def crop_resize(img_file):
        img = cv2.imread(img_file, cv2.IMREAD_COLOR)
        y, x, c = img.shape
        if y < x:
            aux = int((x-y)/2)
            img = img[:,aux:aux+y]
        elif x < y:
            aux = int((y-x)/2)
            img = img[aux:aux+x,:]
        return cv2.resize(img, (224, 224))

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

if __name__ == '__main__':
    img_dict, img_dict_path = load_images_from_file('Images/labels.txt')  # take all images category by category

    gist = gist_img(img_dict)
    name = gist[0]
    desc_list = gist[1] # Takes the descriptor list which is unordered one
    cat_dict = gist[2] # Takes the sift features that is seperated class by class for train data
    create_dir(f"Output/")
    create_dir(f"Output/{name}/")
    save(cat_dict, img_dict_path, f"Output/{name}/")
