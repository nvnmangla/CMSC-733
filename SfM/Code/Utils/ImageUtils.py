import os
from turtle import color
import numpy as np
import cv2


def get_KeyPointMatches(matches_file_path, file_name):
    primary_image_num = int(file_name.split("matching")[-1].split(".txt")[0])
    with open(os.path.join(matches_file_path, file_name)) as raw_data:
        lines = raw_data.readlines()[1:]
        all_coords = list()
        for line in lines:
            elems = line.split()
            num_matches = int(elems[0])
            rgb_vals = (int(elems[1]), int(elems[2]), int(elems[3]))
            uv_coords_primary = (float(elems[4]), float(elems[5]))
            a = 1
            while num_matches > 1:
                img_num = int(elems[5+a])
                uv_coords = (float(elems[6+a]), float(elems[7+a]))
                all_coords.append(np.array([primary_image_num, img_num, uv_coords_primary, uv_coords], dtype=object))
                a += 3
                num_matches -= 1
        all_coords = np.array(all_coords, dtype=object)
    return all_coords


def get_ImageSet(image_path):
    image_list = list()
    for file_name in sorted(os.listdir(image_path)):
        if ".jpg" in file_name:
            image = cv2.imread(os.path.join(image_path, file_name))
            image_list.append(image)
    return image_list


def organize_all_matches(file_dir = "Data/"):
    all_file_paths = list()
    for file_name in sorted(os.listdir(file_dir)):
        if ("matching") in file_name:
            all_file_paths.append(os.path.join(file_dir, file_name))

    color_info = list()
    pixel_matches = list()
    feature_flags = list()
    for i, data_path in enumerate(all_file_paths):
        with open(data_path) as raw_data:
            lines = raw_data.readlines()[1:]
            for line in lines:
                uv_vals = np.empty((1, len(all_file_paths)+1), dtype=object)
                flag_row = np.zeros((1, len(all_file_paths)+1))
                
                elems = line.split()
                n_matches = int(elems[0])
                rgb_vals = (int(elems[1]), int(elems[2]), int(elems[3]))
                color_info.append(rgb_vals)
                u1, v1 = float(elems[4]), float(elems[5])
                
                uv_vals[0, i] = (u1, v1)
                flag_row[0, i] = 1
                a = 1
                while n_matches > 1:
                    image_number = int(elems[5+a])
                    u2, v2 = float(elems[6+a]), float(elems[7+a])
                    uv_vals[0, image_number-1] = (u2, v2)
                    flag_row[0, image_number-1] = 1
                    a += 3
                    n_matches -= 1
                
                pixel_matches.append(uv_vals)
                feature_flags.append(flag_row)
    return np.array(pixel_matches).reshape(-1, len(all_file_paths)+1), np.array(feature_flags, dtype=int).reshape(-1, len(all_file_paths)+1), np.array(color_info)