

"""
CMSC733 Spring 2021: Classical and Deep Learning Approaches for
Geometric Computer Vision
Project1: MyAutoPano: Phase 1 Starter Code

Author(s):
Naveen Mangla (nmangla.umd.edu)
University of Maryland, College Park
"""
import random
import cv2
import numpy as np
import glob
import copy
import argparse
import matplotlib.pyplot as plt

###########   ANMS   ############################


def ANMS(img, best):
    print("Performing ANMS")
    img2 = copy.deepcopy(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    dst = cv2.goodFeaturesToTrack(gray,1000, 0.01,10)
    
    p, _, _ = dst.shape
    print(p, "Features Detected")
    
    ED = 0
    r = np.ones((p, 3))
    r[:, 0] = np.exp(50)
    for i in range(p):
        for j in range(p):
            xi = int(dst[i, :, 0])
            yi = int(dst[i, :, 1])
            xj = int(dst[j, :, 0])
            yj = int(dst[j, :, 1])
            if gray[yi, xi] > gray[yj, xj]:
                ED = (xj-xi)**2 + (yj-yi)**2
            if ED < r[i, 0]:
                r[i, 0] = ED
                r[i, 2] = xi
                r[i, 1] = yi
    result = r[np.argsort(r[:, 0])]
    result = np.flipud(r)[:best, :]
    
    return result

###############   FEATURE DESCRIPTOR   ######################################


def fdis(path, img, best, patch_size):
    print("Describing Features")
    corners = ANMS(img, best)
    X = corners[:, 1:]
    mid = patch_size/2
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sample = []
    points = []
    for i in range(X.shape[0]):
        x0 = X[i, 0]
        y0 = X[i, 1]
        if (mid < x0 < gray.shape[0]-mid and mid < y0 < gray.shape[1]-mid):
            patch = gray[int(x0-mid):int(x0+mid), int(y0-mid):int(y0+mid)]
            dst = cv2.GaussianBlur(patch, (5, 5), cv2.BORDER_DEFAULT)
            res = cv2.resize(
                patch, (8, 8), interpolation=cv2.INTER_LINEAR)

            cv2.imwrite(path+'/Code/solution/feature' +
                        str(i)+'.png', patch)
            res_vector = np.reshape(res, (64, 1))
            res_vector = res_vector-np.mean(res_vector)
            res_vector = res_vector / (np.std(res_vector))
            point = (x0, y0)
            points.append(point)
            sample.append(res_vector)
    return sample, points

#################    CROPING THE FINAL IMAGE  #############################


def crop(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

    contours, hierarchy = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = contours[0]
    x, y, w, h = cv2.boundingRect(cnt)

    crop = img[y:y+h, x:x+w]
    return crop

##############      MATHING FEATURES      #####################


def match_features(path, im1, im2, best):
    size = 41
    fet1, cor1 = fdis(path, im1, best, size)
    fet2, cor2 = fdis(path, im2, best, size)

    if len(fet1) == len(cor1) and len(fet2) == len(cor2):
        p = len(fet1)
        q = len(fet2)
        data = list()
        for i in range(p):
            points = np.zeros((1, 3))
            points[0, 1] = cor1[i][0]
            points[0, 2] = cor1[i][1]
            dsum = np.zeros((1, 3))
            for j in range(q):
                dis = (fet1[i]-fet2[j])**2
                dsum[0, 0] = np.sum(dis)
                dsum[0, 1] = cor2[j][0]
                dsum[0, 2] = cor2[j][1]
                points = np.vstack((points, dsum))
            data.append(points)
        pair_stack = []
        for point in data:
            sorted_dis = point[point[:, 0].argsort()]
            ratio = sorted_dis[1, 0]/sorted_dis[2, 0]
            if ratio < 0.7:
                pair = np.vstack((sorted_dis[0, 1:], sorted_dis[1, 1:]))
                pair_stack.append(pair)
        print("Features Matched =", len(pair_stack))
        if len(pair_stack) < 5:
            print("Very less features,cant match")
            return [], False
        else:
            
            return pair_stack, True


##################################################################


def draw(stack, im1, im2):

    temp = np.zeros((max(im1.shape[0], im2.shape[0]), im1.shape[1] +
                    im2.shape[1], im1.shape[2]), type(im1.flat[0]))
    temp[:im1.shape[0], :im1.shape[1], :] = im1
    temp[:im2.shape[0], im1.shape[1]:, :] = im2
    # temp = cv2.hconcat([im1, im2])
    for i in range(len(stack)):
        x1 = int(stack[i][0, 1])
        y1 = int(stack[i][0, 0])
        x2 = int(stack[i][1, 1] + im1.shape[1])
        y2 = int(stack[i][1, 0])

        cv2.line(temp, (x1, y1), (x2, y2), (0, 255, 255), 1)
        cv2.circle(temp, (x1, y1), 3, 255, -1)
        cv2.circle(temp, (x2, y2), 3, 255, -1)

    return temp

##################################################################


def Homography(path, img1, img2, best, thresh, N):
    pairs, flag = match_features(path, img1, img2, best)
    if flag:
        model, status = RANSAC(pairs, thresh, N)
        im12 = draw(model, img1, img2)
        cv2.imwrite(path+"/Code/solution/RANSAC.png", im12)
        source = np.float32([p[0, :] for p in model])
        dest = np.float32([p[1, :] for p in model])
        L, _ = cv2.findHomography(np.float32(source), np.float32(dest))
        return L, status
    else:
        print("Skipping")
        return [], False


##################################################################

def RANSAC(pairs, thresh, N):
    new = 0
    model = []
    for n in range(N):
        iliners = list()
        pair = list()
        while len(pair) < 4:
            ind = np.random.randint(0, len(pairs))
            if ind not in pair:
                pair.append(ind)
        p1 = np.array([pairs[pair[0]][0, :], pairs[pair[1]][0, :], pairs[pair[2]]
                       [0, :], pairs[pair[3]][0, :]], np.float32)
        p2 = np.array([pairs[pair[0]][1, :], pairs[pair[1]][1, :], pairs[pair[2]]
                       [1, :], pairs[pair[3]][1, :]], np.float32)
        H = cv2.getPerspectiveTransform(p1, p2)
        for i in range(len(pairs)):

            expected = np.dot(H, np.array([pairs[i][0, 0], pairs[i][0, 1], 1]))
            if expected[2] == 0:
                expected[2] = np.exp(-50)
            px = expected[0]/(expected[2])
            py = expected[1]/(expected[2])
            expected = np.array([px, py])
            expected = np.float32([point for point in expected])
            real = np.array([pairs[i][1, 0], pairs[i][1, 1]])
            if np.linalg.norm(expected-real) < thresh:

                iliners.append(i)
        if new < len(iliners):
            new = len(iliners)
            model = [pairs[i] for i in iliners]
    if len(model) < 5:
        print("Not enough RANSAC Match  "+str(len(model))+" found")
        return model, False
    else:
        print(str(len(model))+" found")
        return model, True

##################################################################


def stitch(H, img1, img2):

    print("Stitching")
    p, q, _ = img1.shape
    origin = np.float32([[0, 0, 1], [q, 0, 1], [0, p, 1], [q, p, 1]]).T

    p_m = np.matmul(H, origin)

    px = [p_m[0, i]/p_m[2, i] for i in range(p_m.shape[1])]
    py = [p_m[1, i]/p_m[2, i]for i in range(p_m.shape[1])]
    ymax = np.max(py)
    xmax = np.max(px)
    xmin = np.min(px)
    ymin = np.min(py)
    T = np.float32([[1, 0, -1*xmin], [0, 1, -1*ymin], [0, 0, 1]])
    H = np.dot(T, H)
    shape = (int(xmax-xmin+img1.shape[1]), int(ymax-ymin+img1.shape[0]))
    im3 = cv2.warpPerspective(img1, H, shape)
    for x in range(img2.shape[0]):
        for y in range(img2.shape[1]):
            img2_x = int(x + abs(xmin))
            img2_y = int(y + abs(ymin))
            im3[img2_x, img2_y, :] = img2[x, y, :]
    return crop(im3)
##################################################################

def Blend(path, images, best, thresh, N):
    img1 = images[0]
    
    for img in images[1:2]:
        H, status = Homography(path, img1, img, best, thresh, N)
        if not status:
            print('Cannot Stitch')
            break
        else:
            img1 = stitch(H, img1, img)
            
    return img1

##############################################################
##############################################################
##############################################################


def main():
    Parser = argparse.ArgumentParser()
    Parser.add_argument('--BasePath', default='/home/naveen/CMSC733/nmangla_p1/Phase1/Data/Train/Set1',
                        help='Give your path')

    Args = Parser.parse_args()
    BasePath = Args.BasePath

    images = [cv2.imread(file)
              for file in sorted(glob.glob(str(BasePath)+'/*.jpg'))]

    path = '/home/naveen/CMSC733/nmangla_p1/Phase1'
    # for i in range(len(images)):
    #     img = copy.deepcopy(images[i])
    #     gray = cv2.cvtColor(images[i], cv2.COLOR_BGR2GRAY)
    #     gray = np.float32(gray)
    #     dst = cv2.goodFeaturesToTrack(gray,10000, 0.001,5)
    #     for j in range(dst.shape[0]):
    #         cv2.circle(img, (int(dst[j,:,0]), int(
    #             dst[j,:,1])),1, (0, 0, 255), -1)
    #     cv2.imwrite(path+"/Code/solution/corners_"+str(i+1)+".jpg",img)
    #     img = copy.deepcopy(images[i])
        
    #     result = ANMS(img,500)
    #     x_pts = result[:, 2]
    #     y_pts = result[:, 1]
    #     for j in range(result.shape[0]):
    #         cv2.circle(img, (int(x_pts[j]), int(y_pts[j])), 1, (0, 0, 255), -1)
    #     cv2.imwrite(path+"/Code/solution/ANMS_"+str(i+1)+".png", img)
    #     img2 = copy.deepcopy(images[i+1])

    #     img = copy.deepcopy(images[i])
    #     pair,_ = match_features(path,img,img2,500)
    #     temp  = draw(pair,img,img2)
    #     cv2.imwrite(path+"/Code/solution/match_"+str(i)+"_"+str(i+1)+".png",temp)
        
    #     ran, _ = RANSAC(pair,30,3000)
    #     temp = draw(ran, img, img2)
    #     cv2.imwrite(path+"/Code/solution/ransac_" +
    #                 str(i)+"_"+str(i+1)+".png", temp)
            
        
    mypano = Blend(path, images, best =700, thresh=40, N=4000)
    
    cv2.imwrite(path+"/Code/solution/mypano.png", mypano)


if __name__ == '__main__':
    main()
