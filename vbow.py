import numpy as np
import cv2
import faiss
from utils_gist import *
from extract_patches.core import extract_patches
from util import load_images_from_file, create_dir, save


TEST = False

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

def sift_features(img_dict, kp_dict):
    # Creates descriptors using sift library
    # Takes one parameter that is images dictionary
    # Return an array whose first index holds the decriptor_list without an order
    # And the second index holds the sift_vectors dictionary which holds the descriptors but this is seperated class by class
    sift_dict = dict()
    descriptor_list = list()
    sift = cv2.xfeatures2d.SIFT_create()
    for key,value in img_dict.items():
        features = list()
        for i, img in enumerate(value):
            kp = kp_dict.get(key)[i]
            kp, des = sift.compute(img, kp)
            if des is not None:
                descriptor_list.extend(des)
                features.append(des)
            if TEST and i>5:
                break
        sift_dict[key] = features

    return ["sift", descriptor_list, sift_dict]

def extract_kp(img_dict):
    kp_dict = dict()
    fast = cv2.FastFeatureDetector_create()
    sift = cv2.xfeatures2d.SIFT_create()
    for key,value in img_dict.items():
        kp_list = list()
        for i, img in enumerate(value):

            kpf = fast.detect(img,None)
            print( "Total Keypoints with nonmaxSuppression: {}".format(len(kpf)) )
            img2 = cv2.drawKeypoints(img, kpf, None, color=(255,0,0))
            cv2.imwrite(f'fast_default_{len(kpf)}.jpg',img2)

            fast.setThreshold(20)
            kpf = fast.detect(img,None)
            print( "Total Keypoints with nonmaxSuppression: {}".format(len(kpf)) )
            img2 = cv2.drawKeypoints(img, kpf, None, color=(255,0,0))
            cv2.imwrite(f'fast_t20_{len(kpf)}.jpg',img2)

            fast.setThreshold(30)
            kpf = fast.detect(img,None)
            print( "Total Keypoints with nonmaxSuppression: {}".format(len(kpf)) )
            img2 = cv2.drawKeypoints(img, kpf, None, color=(255,0,0))
            cv2.imwrite(f'fast_t30_{len(kpf)}.jpg',img2)

            kp = sift.detect(img,None)
            print( "Total Keypoints with nonmaxSuppression: {}".format(len(kp)) )
            img2 = cv2.drawKeypoints(img, kp, None, color=(255,0,0))
            cv2.imwrite(f'sift_default_{len(kp)}.jpg',img2)

            mean = np.mean([p.response for p in kp])
            kp_filter = [p for p in kp if p.response > mean]
            print( "Total Keypoints with nonmaxSuppression: {}".format(len(kp_filter)) )
            img2 = cv2.drawKeypoints(img, kp_filter, None, color=(255,0,0))
            cv2.imwrite(f'sift_mean_{len(kp_filter)}.jpg',img2)

            remove_list = list()
            for i, p in enumerate(kp):
                #p.size = p.size * 2
                for j in range (i+1, len(kp)):
                    over = cv2.KeyPoint_overlap(p, kp[j])
                    if over > 0 and not(j in remove_list):
                        remove_list.append(j)
            kp_ol = [p for i, p in enumerate(kp) if i not in remove_list]
            print( "Total Keypoints with nonmaxSuppression: {}".format(len(kp_ol)) )
            img2 = cv2.drawKeypoints(img, kp_ol, None, color=(255,0,0))
            cv2.imwrite(f'sift_ol_{len(kp_ol)}.jpg',img2)

            remove_list = list()
            for i, p in enumerate(kp_filter):
                #p.size = p.size * 2
                for j in range (i+1, len(kp_filter)):
                    over = cv2.KeyPoint_overlap(p, kp_filter[j])
                    if over > 0 and not(j in remove_list):
                        remove_list.append(j)
            kp_ol = [p for i, p in enumerate(kp_filter) if i not in remove_list]
            print( "Total Keypoints with nonmaxSuppression: {}".format(len(kp_ol)) )
            img2 = cv2.drawKeypoints(img, kp_ol, None, color=(255,0,0))
            cv2.imwrite(f'sift_mean_ol_{len(kp_ol)}.jpg',img2)

            remove_list = list()
            for i, p in enumerate(kp_filter):
                p.size = p.size * 2
                for j in range (i+1, len(kp_filter)):
                    over = cv2.KeyPoint_overlap(p, kp_filter[j])
                    if over > 0 and not(j in remove_list):
                        remove_list.append(j)
            kp_ol = [p for i, p in enumerate(kp_filter) if i not in remove_list]
            print( "Total Keypoints with nonmaxSuppression: {}".format(len(kp_ol)) )
            img2 = cv2.drawKeypoints(img, kp_ol, None, color=(255,0,0))
            cv2.imwrite(f'sift_mean_2ol_{len(kp_ol)}.jpg',img2)

            # pt_list = list()
            # for p in kp:
            #     pt_list.append(p.pt)
            # x = np.array(pt_list)
            # shp = x.shape[1]
            # kmeans = faiss.Kmeans(shp, 100, niter=10, verbose=True, nredo=1)
            # kmeans.train(x)

            if kp_filter is not None:
                kp_list.append(kp_filter)
            if TEST and i>5:
                break
        kp_dict[key] = kp_list
    return kp_dict

def gist_features(img_dict, kp_dict):
    # Creates descriptors using sift library
    # Takes one parameter that is images dictionary
    # Return an array whose first index holds the decriptor_list without an order
    # And the second index holds the sift_vectors dictionary which holds the descriptors but this is seperated class by class
    patch_size = 64
    mr_size = 1.0
    gist_dict = dict()
    descriptor_list = list()
    gist = GistUtils()
    for key,value in img_dict.items():
        features = list()
        for i, img in enumerate(value):
            kp = kp_dict.get(key)[i]
            patches = extract_patches(kp, img, patch_size, mr_size, 'cv2')
            des = list()
            for sub_img in patches:
                des.append(gist.get_gist_vec(sub_img))
            if des is not None:
                descriptor_list.extend(des)
                features.append(des)
            if TEST and i>5:
                break
        gist_dict[key] = features


    return ["gist", np.float32(descriptor_list), gist_vectors]

def vbow(input_file, output_folder, test=False):
    globals_list = globals()
    globals_list['TEST'] = test
    img, img_path = load_images_from_file(input_file)  # take all images category by category
    size = [50, 100, 250, 500, 750, 1000]
    kp = extract_kp(img)
    gist = gist_features(img, kp)
    sift = sift_features(img, kp)
    descriptors = [gist, sift]

    for d in descriptors:
        name = d[0]
        desc_list = d[1] # Takes the descriptor list which is unordered one
        cat_list = d[2] # Takes the sift features that is seperated class by class for train data
        create_dir(output_folder)
        create_dir(f"{output_folder}/{name}/")
        for s in size:
            create_dir(f"{output_folder}/{name}/{s}/")
        for s in size:
            vbow = generate_vbow(s, desc_list, cat_list, 100, 10) # Takes the central points which is visual words
            save(vbow, img_path, f"{output_folder}/{name}/{s}/")

if __name__ == '__main__':
    vbow('Images/labels.txt', "Output", True)

