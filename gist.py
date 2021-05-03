import numpy as np
from utils_gist import *
from util import load_images_from_file, create_dir, save


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
            print(f"Generating GIST descriptors: {i}")
            des = gist.get_gist_vec(img)
            if des is not None:
                descriptor_list.extend(des)
                features.append(des)
        gist_vectors[key] = np.float32(features)

    return ["gist", np.float32(descriptor_list), gist_vectors]

def gist(input_file, output_folder):

    img_dict, img_dict_path = load_images_from_file(input_file)  # take all images category by category

    gist = gist_img(img_dict)
    name = gist[0]
    desc_list = gist[1] # Takes the descriptor list which is unordered one
    cat_dict = gist[2] # Takes the sift features that is seperated class by class for train data
    create_dir(output_folder)
    create_dir(f"{output_folder}/{name}/")
    save(cat_dict, img_dict_path, f"{output_folder}/{name}/")
    print(f"\nThe execution of gist is done!\n")

if __name__ == '__main__':
    gist('Images/labels_small.txt', "Output", True)

