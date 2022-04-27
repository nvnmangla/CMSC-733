from statistics import mean
from black import err
import cv2
import numpy as np
import argparse
import glob
import matplotlib.pyplot as plt
import math
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
from sympy import unflatten

def v(i,j,H):
    
    i = i-1
    j = j-1
   
    vij = [H[0, i]*H[0, j], H[0, i]*H[1, j] + H[1, i]
           * H[0, j], H[1, i]*H[1, j], H[0, i]*H[2, j] + H[2, i]*H[0, j], H[1, i]*H[2, j] + H[2, i]*H[1, j], H[2, i]*H[2, j]]
    
    return np.array(vij).reshape(6,1)

def projection(K,H):
 
    l1= np.linalg.norm(np.matmul(np.linalg.inv(K),H[:,0]))
    l2 = np.linalg.norm(np.matmul(np.linalg.inv(K), H[:, 1]))
    l = (np.linalg.norm(l1)+np.linalg.norm(l2))/2
    lr1r2t =  np.matmul(np.linalg.inv(K),H)

    sgn = np.linalg.det(lr1r2t)
    if sgn<0:
        r = lr1r2t*-1/l
    elif sgn>=0:
        r = lr1r2t/l
    # r = lr1r2t/l
    r1 = r[:,0]
    r2 = r[:,1]

    r3 = np.cross(r1,r2)
    t = r[:,2]

    Q = np.array([r1,r2,r3]).T
    u,s,v =np.linalg.svd(Q)
    R = np.matmul(u,v)


    temp = np.hstack((R,t.reshape(3,1)))
    return temp


def param(A,k1,k2):
    a = np.reshape(np.array([A[0][0],0,A[0][2],A[1][1],A[1][2]]),(5,1))
    
    
    b = np.array([k1,k2]).reshape(2,1)
    p = np.concatenate([b,a])
    return p

def function(params,corner_list,H_list):
     
    A = np.zeros((3,3))
    A[0,:] = params[2:5]
    A[1,:] = [0,params[5],params[6]]
    A[2,:] = [0,0,1]
    
    
    K = np.reshape(params[:2],(2,1))

    world = []
    for i in range(6):
        for j in range(9):
    
            world.append([21.5*(j+1),21.5*(i+1),0,1])
    world = np.array(world)

    error=np.empty([54,1])
    for ind in range(len(H_list)):
        H = H_list[ind]
        corners = corner_list[ind]
        
        R = projection(A,H) 
        new_world = np.matmul(R,world.T)
        new_world = new_world/new_world[2]
        P = np.matmul(A,R)
        imgpt = np.matmul(P,world.T)
        imgpt = imgpt/imgpt[2]

        u0,v0 = A[0,2],A[1,2]

        u,v = imgpt[0],imgpt[1]

        x,y = new_world[0],new_world[1]

        k1,k2 = K[0],K[1]

        u_hat = u+(u-u0)*(k1*(x**2+y**2)+k2*(x**2+y**2)**2)
        v_hat = v+(v-v0)*(k1*(x**2+y**2)+k2*(x**2+y**2)**2)
        
        
        proj = corners
        proj = np.reshape(proj,(-1,2))
        
        reproj = np.reshape(np.array([u_hat,v_hat]),(2,54)).T
        
        err = np.linalg.norm(np.subtract(proj,reproj),axis=1)

        error=np.vstack((error,err.reshape((54,1))))
    error=error[54:]
    error=np.reshape(error,(702,))
    
    return error


def meanerror(A,K,H_list,corner_list):

    world = []
    for i in range(6):
        for j in range(9):
    
            world.append([21.5*(j+1),21.5*(i+1),0])
    world = np.array(world)
    mean = 0
    error = np.zeros([2,1])
    for i in range(len(H_list)):

        lr1r2t = projection(A,H_list[i])

        img_points,_ = cv2.projectPoints(world,lr1r2t[:,0:3],lr1r2t[:,3],A,K)
        img_points = np.array(img_points)
        errors = np.linalg.norm(np.subtract(img_points[:,0,:],corner_list[i][:,0,:]),axis=1)
        error = np.concatenate([error,np.reshape(errors,(errors.shape[0],1))])
    error =np.mean(error)

        
       
    return error



##############################################
    
path = "/home/naveen/CMSC733/AutoCalib/" ############# Change Path 


################################################
Parser = argparse.ArgumentParser()
Parser.add_argument('--BasePath', default=path+"/Calibration_Imgs",
                    help='Give your path')

Args = Parser.parse_args()
BasePath = Args.BasePath

images = [cv2.imread(file) for file in sorted(glob.glob(str(BasePath)+'/*.jpg'))]


V = np.ones((1,6))

H_list = []
corner_list = []

for img in images:
    
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    
    ret, corners = cv2.findChessboardCorners(gray,(9, 6), None)
    
    corner_list.append(corners)
     
        
    camera = []

    
    for k in [0,8,53,45]:  
        camera.append(corners[k][0])
    
        cv2.circle(img, (int(corners[k][0][0]),int(corners[k][0][1])), 10, (255, 0, 0),2)
    
    
    camera  = np.float32(camera)
    
    
    world = np.float32([(21.5, 21.5), (21.5*9, 21.5), (21.5*9, 6*21.5), (21.5, 6*21.5)])
    
    H,_= cv2.findHomography(world, camera)
    H_list.append(H)
     
    V1 = np.vstack((v(1,2,H).T,(v(1,1,H)-v(2,2,H)).T))
   
    V = np.vstack((V,V1))
    
   
     

U,S,L = np.linalg.svd(V[1:]) 

b = L[:][5]
[B11,B12,B22,B13,B23,B33] = b



v0 = (B12*B13-B11*B23)/(B11*B22-B12**2)

l = B33-(B13**2+v0*(B12*B13-B11*B23))/B11    


alpha = math.sqrt(l/B11)

beta = math.sqrt(l*B11/(B11*B22-B12**2))


gama = -1*B12*(alpha**2)*beta/l

  
u0 = gama*v0/beta - B13*(alpha**2)/l


A = np.array([[alpha,gama,u0],[0,beta,v0],[0,0,1]])
origA = A


print("K = ",A)
      

initial = param(A,0,0)
res = least_squares(function,x0=np.squeeze(initial),method='lm',args=(corner_list,H_list))
A = np.zeros((3,3))
A[0,:] = res.x[2:5]
A[1,:] = [0,res.x[5],res.x[6]]
A[2,:] = [0,0,1]
temp = projection(A,H)

print("K after distortion = ",A)
K = res.x[:2]
print("k1,k2 = ",K)

distortion = np.array([K[0],K[1],0,0,0],dtype=float)

world = []
for i in range(6):
    for j in range(9):

        world.append([21.5*(j+1),21.5*(i+1),0,1])
world = np.array(world)

for ind in range(len(images)):
    undist = cv2.undistort(images[ind],A,distortion)
    H = H_list[ind]
    meaner = meanerror(A,distortion,H_list,corner_list)
    corners = corner_list[ind]
    gray = cv2.cvtColor(undist,cv2.COLOR_BGR2GRAY)

    R = projection(A,H) 
    new_world = np.matmul(R,world.T)
    new_world = new_world/new_world[2]
    P = np.matmul(A,R)
    imgpt = np.matmul(P,world.T)
    corners2 = imgpt/imgpt[2]
   
    
    
    

    for k in range(54):  
        cv2.circle(undist, (round(corners2[0,k]),round(corners2[1,k])), 20, (255, 0,0),1)
        
        cv2.circle(undist, (round(corners[k][0][0]),round(corners[k][0][1])), 20, (0, 0, 255),1)


    cv2.imwrite(path+str(ind)+".png",undist)

    
print("Mean Error= ",meaner)
    
    