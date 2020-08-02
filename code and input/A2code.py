import cv2
import numpy
import math
import sys

## for adaptive maximum supression
def adaptive_sus(points,resize):
    sorted_points=list()
    for point_1 in points:
        small_dist=sys.float_info.max
        for point_2 in points:
            distance=(((point_1.pt[0]-point_2.pt[0])**2)+((point_1.pt[1]-point_2.pt[1])**2))**0.5
            if(distance!=0) and point_1.response<(0.9*point_2.response) and distance<small_dist:
                small_dist=distance
        point_1.size=small_dist
        sorted_points.append(point_1)
    sorted_points=sorted(sorted_points, key=lambda point: point.size, reverse=True)
    if len(sorted_points)>resize:
        sorted_points=sorted_points[:resize]
    return sorted_points
	

## for rotation invariance
def rotateinvariance(mag,angle):
    histogram=numpy.zeros(36)
    ret_angles=list()
    for row in range(0,angle.shape[0]):
        for col in range(0,angle.shape[1]):
            angle[row][col]%=360
            key=int(math.floor(angle[row][col]/10))
            histogram[key]+=mag[row][col]
    maxval=max(histogram)
    dst=list()
    for loop in range(0,len(histogram)):
        if histogram[loop]>=(maxval*0.8):
            dst.append(loop)
    for loc in dst:
        new_angle=angle
        for row in range(0,new_angle.shape[0]):
            for col in range(0,new_angle.shape[1]):
                new_angle[row][col]-=(loc*10)
                angle[row][col]%=360
        ret_angles.append(new_angle)
    return ret_angles

## to match features
def create_matchings(pts_1, pts_2):
    final_features1=list()
    final_features2=list()
    distances=list()
    for img1 in pts_1:
        feat_desp1=pts_1[img1]
        totals_pt=dict()
        totals=list()
        for img2 in pts_2:
            total=0
            feat_desp2=pts_2[img2]
            ft_desp=(feat_desp1-feat_desp2)**2
            for loop in ft_desp:
                total+=loop
            
            totals.append(total)
            totals_pt[img2]=total
        totals=sorted(totals)
        
        if totals[0]<0.5 and (totals[0]/totals[1])<0.6: 
            final_features1.append(cv2.KeyPoint(int(img1.split()[0]),int(img1.split()[1]),1))
            ans=""
            for key in totals_pt:
                if totals_pt[key]==totals[0]:
                    ans=key
            final_features2.append(cv2.KeyPoint(int(ans.split()[0]),int(ans.split()[1]),1))
            distances.append(totals[0])
    
	## remove duplicates
    loop1=0
    removeEle=set()
    for key1 in final_features1:
        loop2=0
        key1=key1.pt
        for key2 in final_features1:
            key2=key2.pt
            if loop1!=loop2 and key1[0]==key2[0] and key1[1]==key2[1]:
                if distances[loop1]>distances[loop2]:
                    removeEle.add(loop1)
                else:
                    removeEle.add(loop2)
            loop2+=1
        loop1+=1
    
    loop1=0
    for key1 in final_features2:
        loop2=0
        key1=key1.pt
        for key2 in final_features2:
            key2=key2.pt
            if loop1!=loop2 and key1[0]==key2[0] and key1[1]==key2[1]:
                if distances[loop1]>distances[loop2]:
                    removeEle.add(loop1)
                else:
                    removeEle.add(loop2)
            loop2+=1
        loop1+=1
    
    res_features1=list()
    res_features2=list()
    matchings=list()
    out_loop=0
    for loop in range(0,len(distances)):
        if loop not in removeEle:
            res_features1.append(final_features1[loop])
            res_features2.append(final_features2[loop])
            matchings.append(cv2.DMatch(out_loop,out_loop,distances[loop]))
            out_loop+=1
    
    return res_features1, res_features2, matchings


## to create 16x16 magnitude and angle matrix
def create_mag_angle(inp_img):
    rows, cols=inp_img.shape
    inp_img = numpy.float32(inp_img)
    g_x=numpy.zeros(inp_img.shape)
    g_y=numpy.zeros(inp_img.shape)
    for row in range(1,rows-1):
        for col in range(1,cols-1):
            g_x[row][col]=inp_img[row][col+1]-inp_img[row][col-1]
            g_y[row][col]=inp_img[row+1][col]-inp_img[row-1][col]
            
    magnitude=((g_x**2)+(g_y**2))**0.5
    degrees=numpy.degrees(numpy.arctan2(g_y,g_x))
    return magnitude, degrees
	

## sift working
def sift(input_img, points):
    features=dict()
    grey_inp = cv2.GaussianBlur(cv2.cvtColor(input_img,cv2.COLOR_BGR2GRAY),(0,0),1.5)
    magnitude,degrees=create_mag_angle(grey_inp)
    key_loop=0
    for point in points:
        pt_c=int(point.pt[0])
        pt_r=int(point.pt[1])
        if pt_r-8>=0 and pt_c-8>=0 and pt_r+8<=magnitude.shape[0] and pt_c+8<=magnitude.shape[1]:
            mag16=magnitude[pt_r-8:pt_r+8,pt_c-8:pt_c+8]
            mag16=cv2.normalize(mag16,None,norm_type=cv2.NORM_L2)
            angleout=degrees[pt_r-8:pt_r+8,pt_c-8:pt_c+8]
            array=[0,4,8,12]
            ret_angles=rotateinvariance(mag16,angleout)
            ##multiple keypoints
            for angle16 in ret_angles:
                descrip_128=list()
                for r in array:
                    for c in array:
                        histogram=numpy.zeros(8)
                        window_mag=mag16[r:r+4,c:c+4]
                        window_angle=angle16[r:r+4,c:c+4]
                        for i in range(0,4):
                            for j in range(0,4):
                                window_angle[i][j]%=360
                                key=int(math.floor(window_angle[i][j]/45))
                                histogram[key]+=window_mag[i][j]

                        descrip_128.extend(list(histogram))
                ##normalise descriptor
                descrip_128 = numpy.clip(descrip_128, a_min=0,a_max=0.2)
                descrip_128= numpy.array(descrip_128)
                descrip_128 = cv2.normalize(descrip_128, None, norm_type=cv2.NORM_L2)
                keyName=str(pt_c)+" "+str(pt_r)+" "+str(++key_loop)
                features[keyName]=descrip_128
       
    return features
	

## for local maximal supression
def max_suppression(input_img,threshold):
    rows,cols=input_img.shape
    for row in range(0,rows-3):
        for col in range(0,cols-3):
            a_max=input_img[row:row+3,col:col+3]
            _,max_val,_,(loc_c,loc_r)=cv2.minMaxLoc(a_max)
            if max_val>threshold:
                input_img[row:row+3,col:col+3]=0
                input_img[row+loc_r][col+loc_c]=max_val
            
    return input_img
	

## for corner detection
def harris_points(input_img,smallval,threshold,resize):
    grey_inp=cv2.cvtColor(input_img,cv2.COLOR_BGR2GRAY)
    ix=cv2.Sobel(grey_inp,cv2.CV_32F,1,0,ksize=5)
    iy=cv2.Sobel(grey_inp,cv2.CV_32F,0,1,ksize=5)
    Ix2=ix**2
    gx2=cv2.GaussianBlur(Ix2,(3,3),0)
    Iy2=iy**2
    gy2=cv2.GaussianBlur(Iy2,(3,3),0)
    Ixy=ix*iy
    gxy=cv2.GaussianBlur(Ixy,(3,3),0)
    I_x2_paded=cv2.copyMakeBorder(gx2, 2, 2, 2, 2, cv2.BORDER_CONSTANT, 0)
    I_y2_paded=cv2.copyMakeBorder(gy2, 2, 2, 2, 2, cv2.BORDER_CONSTANT, 0)
    I_xy_paded=cv2.copyMakeBorder(gxy, 2, 2, 2, 2, cv2.BORDER_CONSTANT, 0)
    corner_mat=numpy.zeros(Ixy.shape)
    ret_points=list()
    for row in range(2,I_x2_paded.shape[0]-2):
        for col in range(2,I_x2_paded.shape[1]-2):
            sum_x2=numpy.sum(I_x2_paded[row-2:row+3,col-2:col+3])
            sum_y2=numpy.sum(I_y2_paded[row-2:row+3,col-2:col+3])
            sum_xy=numpy.sum(I_xy_paded[row-2:row+3,col-2:col+3])
            
            determinant=(sum_x2*sum_y2)-(sum_xy**2)
            trace=sum_x2+sum_y2
            corner=determinant/(smallval+trace)
            
            if corner>threshold:
                corner_mat[row-2][col-2]=corner
            
    paded_corner=cv2.copyMakeBorder(corner_mat, 1, 1, 1, 1, cv2.BORDER_CONSTANT, 0)
    paded_corner=max_suppression(paded_corner,threshold)
    for row in range(0,paded_corner.shape[0]):
        for col in range(0,paded_corner.shape[1]):
            if paded_corner[row][col]>threshold:
                keyPt=cv2.KeyPoint(col-1,row-1,1)
                keyPt.response=paded_corner[row][col]
                ret_points.append(keyPt)
                
    return adaptive_sus(ret_points,resize)
    

## to get image features
def image_features(threshold, small_val, adapt_resize, imgName, var, outname):
    img=cv2.imread(imgName)
    points=harris_points(img,small_val,threshold,adapt_resize)
    features=sift(img,points)
    out_img=cv2.drawKeypoints(img, points, None, color=(0,255,0), flags=0)
    cv2.imwrite(outname+".png",out_img)
    return img, features
	

def a2start():
	## constants
	threshold=20000000
	small_val=0.000000000000000001
	adapt_resize=500
	print("****STEP1****")
	box="given_images/Boxes.png"
	box, features1=image_features(threshold, small_val, adapt_resize, box, "1", "1a")
	print("Harris output for Boxes image saved as 1a.png")
	
	img1="given_images/Rainier1.png"
	img1, features1=image_features(threshold, small_val, adapt_resize, img1, "1", "1b")
	print("Harris output for Rainier1 image saved as 1b.png")
	
	img2="given_images/Rainier2.png"
	img2, features2=image_features(threshold, small_val, adapt_resize, img2, "2", "1c")
	print("Harris output for Rainier2 image saved as 1c.png")
	
	print("\n****STEP2****")
	
	features1, features2, matchings=create_matchings(features1, features2)
	out_img=cv2.drawMatches(img1,features1,img2,features2, matchings, None, flags=2)
	cv2.imwrite("2.png",out_img)
	print("Matchings output for Rainier1 and Rainier2 image saved as 2.png")
	
