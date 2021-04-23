import os
import numpy as np
import cv2
import faiss
from utils_gist import *

TEST = False

def create_dir(dir):
    if not os.path.isdir(dir):
            try:
                os.mkdir(dir)
            except:
                print(f"Creation of directory {dir} failed!")
                exit(1)

def save(vbow, cat_path_list, output_folder):
    for cat,img_list in vbow.items():
        for i, hist in enumerate(img_list):
            path_list = cat_path_list.get(cat)
            path = path_list[i]
            folder_tree = path.split(".")[0]
            file_name = folder_tree.split("/")[-1]
            output_file = f"{output_folder}/{file_name}.txt"
            np.savetxt(output_file, hist)

def generate_vbow(k, descriptor_list, cat_list, niter=100, nredo=10):
    # A k-means clustering algorithm who takes 2 parameter which is number of cluster(k) and the other is descriptors list(unordered 1d array)
    # Returns an array that holds central points.
    if TEST:
        k = 5
        niter = 10
        nredo = 1
    x = np.array(descriptor_list)
    shp = x.shape[1]
    kmeans = faiss.Kmeans(shp, k, niter=niter, verbose=True, nredo=nredo)
    kmeans.train(x)

    # Takes 2 parameters. The first one is a dictionary that holds the descriptors that are separated class by class
    # And the second parameter is an array that holds the central points (visual words) of the k means clustering
    # Returns a dictionary that holds the histograms for each images that are separated class by class.
    dict_vbow = {}
    for cat_key,desc_list in cat_list.items():
        cat_list = []
        for d in desc_list:
            histogram = np.zeros(k)
            #To compute the mapping from a set of vectors x to the cluster centroids after kmeans has finished training, use:
            dist, ind = kmeans.index.search(np.array(d), 1)
            for i in ind:
                histogram[i] += 1
            cat_list.append(histogram)
        dict_vbow[cat_key] = cat_list
    return dict_vbow

def sift_features(images):
    # Creates descriptors using sift library
    # Takes one parameter that is images dictionary
    # Return an array whose first index holds the decriptor_list without an order
    # And the second index holds the sift_vectors dictionary which holds the descriptors but this is seperated class by class
    sift_vectors = {}
    descriptor_list = []
    sift = cv2.xfeatures2d.SIFT_create()
    for key,value in images.items():
        features = []
        for i, img in enumerate(value):
            kp, des = sift.detectAndCompute(img,None)
            if des is not None:
                descriptor_list.extend(des)
                features.append(des)
                sift_vectors[key] = features
            if TEST and i>10:
                break

    return ["sift", descriptor_list, sift_vectors]

def gist_features(images):
    # Creates descriptors using sift library
    # Takes one parameter that is images dictionary
    # Return an array whose first index holds the decriptor_list without an order
    # And the second index holds the sift_vectors dictionary which holds the descriptors but this is seperated class by class
    gist_vectors = {}
    descriptor_list = []
    gist = GistUtils()
    for key,value in images.items():
        features = []
        for i, img in enumerate(value):
            des = gist.get_gist_vec(img)
            if des is not None:
                descriptor_list.extend(des)
                features.append(des)
                gist_vectors[key] = np.float32(features)
            if TEST and i>10:
                break

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
    img, img_path = load_images_from_file('Images/labels.txt')  # take all images category by category
    size = [50, 100, 250, 500, 750, 1000]
    gist = gist_features(img)
    sift = sift_features(img)
    descriptors = [gist, sift]

    for d in descriptors:
        name = d[0]
        desc_list = d[1] # Takes the descriptor list which is unordered one
        cat_list = d[2] # Takes the sift features that is seperated class by class for train data
        create_dir(f"Output/")
        create_dir(f"Output/{name}/")
        for s in size:
            create_dir(f"Output/{name}/{s}/")
        for s in size:
            vbow = generate_vbow(s, desc_list, cat_list, 100, 10) # Takes the central points which is visual words
            save(vbow, img_path, f"Output/{name}/{s}/")
