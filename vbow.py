import numpy as np
import cv2
import faiss
from utils_gist import *
from extract_patches.core import extract_patches
from util import load_images_from_file, create_dir, save


def generate_vbow(k, descriptor_list, cat_dict, niter=100, nredo=10):
    # A k-means clustering algorithm who takes 2 parameter which is number of cluster(k) and the other is descriptors list(unordered 1d array)
    # Returns an array that holds central points.
    print(f"--- Creating VBoW with {k} words from a list of {len(descriptor_list)}---")
    x = np.array(descriptor_list)
    shp = x.shape[1]
    kmeans = faiss.Kmeans(shp, k, niter=niter, verbose=True, nredo=nredo)
    kmeans.train(x)

    # Takes 2 parameters. The first one is a dictionary that holds the descriptors that are separated class by class
    # And the second parameter is an array that holds the central points (visual words) of the k means clustering
    # Returns a dictionary that holds the histograms for each images that are separated class by class.
    vbow_dict = dict()
    for cat_key,desc_list in cat_dict.items():
        cat_list = list()
        for d in desc_list:
            histogram = np.zeros(k)
            if d is not None:
                #To compute the mapping from a set of vectors x to the cluster centroids after kmeans has finished training, use:
                try:
                    dist, ind = kmeans.index.search(d, 1)
                    for i in ind:
                        histogram[i] += 1
                except Exception as e_d:
                    print(f"Erro ao processar d: {d}")
            cat_list.append(histogram)
        vbow_dict[cat_key] = cat_list
    return vbow_dict

def extract_kp(img_dict):

    def filter_response_size(kp):
        mean_r = np.mean([p.response for p in kp])
        mean_s = np.mean([p.size for p in kp])
        kp_rs = [p for p in kp if (p.size > mean_s) and (p.response > mean_r)]
        return kp_rs

    def filter_overlap(kp):
        remove_list = list()
        for i, p in enumerate(kp):
            for j in range (i+1, len(kp)):
                over = cv2.KeyPoint_overlap(p, kp[j])
                if over > 0 and not(j in remove_list):
                    remove_list.append(j)
        kp_ol = [p for i, p in enumerate(kp) if i not in remove_list]
        return kp_ol

    print("--- Extracting KEYPOINTS from images. ---")
    kp_dict = dict()
    sift = cv2.xfeatures2d.SIFT_create()
    for key,value in img_dict.items():
        kp_list = list()
        for i, img in enumerate(value):

            kp = sift.detect(img,None)
            kp = filter_response_size(kp)
            kp = filter_overlap(kp)

            if kp is not None:
                kp_list.append(kp)
        kp_dict[key] = kp_list
    return kp_dict

def sift_features(img_dict, kp_dict):
    # Creates descriptors using sift library
    # Takes one parameter that is images dictionary
    # Return an array whose first index holds the decriptor_list without an order
    # And the second index holds the sift_vectors dictionary which holds the descriptors but this is seperated class by class
    print("--- Extracting SIFT features from images. ---")
    sift_dict = dict()
    descriptor_list = list()
    sift = cv2.xfeatures2d.SIFT_create()
    cont = 1
    for key,value in img_dict.items():
        features = list()
        for i, img in enumerate(value):
            kp = kp_dict.get(key)[i]
            kp, des = sift.compute(img, kp)
            if des is not None:
                descriptor_list.extend(des)
            features.append(des)
            print(f"\nSIFT descriptors for the {cont} image successfully generated.\n")
            cont += 1
        sift_dict[key] = features
    return ["sift", descriptor_list, sift_dict]

def gist_features(img_dict, kp_dict):
    # Creates descriptors using sift library
    # Takes one parameter that is images dictionary
    # Return an array whose first index holds the decriptor_list without an order
    # And the second index holds the sift_vectors dictionary which holds the descriptors but this is seperated class by class
    print("--- Extracting GIST features from images. ---")
    patch_size = 64
    mr_size = 1.0
    gist_dict = dict()
    descriptor_list = list()
    gist = GistUtils()
    cont = 1
    for key,value in img_dict.items():
        features = list()
        for i, img in enumerate(value):
            kp = kp_dict.get(key)[i]
            patches = extract_patches(kp, img, patch_size, mr_size, 'cv2')
            des = list()
            for j, sub_img in enumerate(patches):
                des_patch = gist.get_gist_vec(sub_img)[0]
                des.append(des_patch)
                print(f"GIST descriptors for the patch {j}/{len(patches)} from image {cont} generated.\n")
            if des is not None:
                des = np.array(des, dtype="float32")
                descriptor_list.extend(des)
            features.append(des)
            print(f"\nGIST descriptors for the {cont} image successfully generated.\n")
            cont += 1
        gist_dict[key] = features


    return ["gist", np.float32(descriptor_list), gist_dict]

def vbow(input_file, output_folder):
    img, img_path = load_images_from_file(input_file)  # take all images category by category
    size = [5, 50, 100, 250, 500, 750, 1000]
    kp = extract_kp(img)
    #sift = sift_features(img, kp)
    gist = gist_features(img, kp)
    descriptors = [gist]
    #descriptors = [gist, sift]

    for d in descriptors:
        name = d[0]
        desc_list = d[1] # Takes the descriptor list which is unordered one
        cat_list = d[2] # Takes the sift features that is seperated class by class for train data
        create_dir(output_folder)
        create_dir(f"{output_folder}/{name}/")
        for s in size:
            create_dir(f"{output_folder}/{name}/{s}/")
        print(f"--- Create VBoW for {name} features. ---")
        for s in size:
            vbow = generate_vbow(s, desc_list, cat_list, 100, 10) # Takes the central points which is visual words
            save(vbow, img_path, f"{output_folder}/{name}/{s}/")

if __name__ == '__main__':
    vbow('Images/labels.txt', "Output")

