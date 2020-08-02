import cv2
import numpy
import random
import math
from A2code import a2start

	
##########################################################  Step 2 - find descriptors and draw matches
def find_matches(inp_img1,inp_img2):
	sift = cv2.xfeatures2d.SIFT_create()
	
	##points and descriptors for both images
	insPts1, despImg1 = sift.detectAndCompute(inp_img1, None)
	insPts2, despImg2 = sift.detectAndCompute(inp_img2, None)
	
	bfmatcher = cv2.BFMatcher(crossCheck=True)
	mts = sorted(bfmatcher.match(despImg1,despImg2), key = lambda loop:loop.distance)
	
	return mts, insPts1, insPts2
	
##########################################################  Step 3 - RANSAC
## get first second points wrt matches
def get_first_second(mts, insPts1, insPts2):
	first = list()
	second  = list()
	for mt in mts:
		first.append(insPts1[mt.queryIdx].pt)
		second.append(insPts2[mt.trainIdx].pt)
		
	return numpy.float32(first), numpy.float32(second)
	
## project point using the homography
def project(col1, row1, hom):
	try:
		add = 0.000000000000000000000001
		out = numpy.dot(hom, numpy.array([col1, row1, 1], dtype=numpy.float32))
		col2, row2 = out[0] / (out[2]+add), out[1] / (out[2]+add)
		return col2, row2
	except:
		return None, None
		
## helper function for RANSAC that computes the number of inlying points given a homography 
def computeInlierCount(hom, matches, first, second, thresh):
	inliers_match=list()
	for loop in range(0, len(matches)):
		project_col2, project_row2 = project(first[loop][0], first[loop][1], hom)
		if project_col2 is None or project_row2 is None:
			continue
		col_dist = (project_col2 - second[loop][0])**2
		row_dist = (project_row2 - second[loop][1])**2
		
		##distance is less than inlier threshold given
		if math.sqrt(col_dist + row_dist) < thresh:
			inliers_match.append(matches[loop])
		
	return len(inliers_match), inliers_match

##  returns the homography transformation for a list of matches between two images
def RANSAC (matches , numIterations, inlierThresh, insPts1, insPts2):
	if len(matches)<4:
		return None, None, None
		
	best_hom = None
	count = 0
	first, second = get_first_second(matches, insPts1, insPts2)
	
	for loop in range(0, numIterations):
		fourMts = random.sample(matches, 4)
		four1, four2 = get_first_second(fourMts, insPts1, insPts2)
		
		## homography and inliers for 4 matches
		hom, _ = cv2.findHomography(four1, four2, 0)
		innerCount, _ = computeInlierCount(hom, matches, first, second, inlierThresh)
		if innerCount > count:
			count = innerCount
			best_hom = hom
	
	## inliers for best homography	
	_, inliers_mts = computeInlierCount(best_hom, matches, first, second, inlierThresh)
	
	best1, best2 = get_first_second(inliers_mts, insPts1, insPts2)
	hom, _ = cv2.findHomography(best1, best2, 0)
	
	return inliers_mts, hom, numpy.linalg.inv(hom)

##########################################################  Step 4 - Stitching
##to get the size of the stitched image by projecting corners of image2
def get_corners(inp_img1, inp_img2, hom, hom_inverse):
	img1_rows, img1_cols, _=inp_img1.shape
	img2_rows, img2_cols, _=inp_img2.shape
	
	project_lt=[int(round(val, 0)) for val in project(0, 0, hom_inverse)]
	project_lt=[project_lt[1], project_lt[0]]
	project_lb=[int(round(val, 0)) for val in project(0, img2_rows-1, hom_inverse)]
	project_lb=[project_lb[1], project_lb[0]]
	project_rt=[int(round(val, 0)) for val in project(img2_cols-1, 0, hom_inverse)]
	project_rt=[project_rt[1], project_rt[0]]
	project_rb=[int(round(val, 0)) for val in project(img2_cols-1, img2_rows-1, hom_inverse)]
	project_rb=[project_rb[1], project_rb[0]]
	
	left_col=min(project_lt[1], project_lb[1], 0)
	right_col=max(project_rt[1], project_rb[1], inp_img1.shape[1]-1)
	top_row=min(project_lt[0], project_rt[0], 0)
	bottom_row=max(project_lb[0], project_rb[0], inp_img1.shape[0]-1)
	
	total_cols=right_col-left_col+1
	total_rows=bottom_row-top_row+1
	
	col_add=left_col if left_col>=0 else left_col*(-1)
	row_add=top_row if top_row>=0 else top_row*(-1)
			
	return total_rows, total_cols, row_add, col_add

## to produce a stitched image
def stitch(inp_img1, inp_img2, hom, hom_inverse):
	stitched_rows, stitched_cols, row_add, col_add=get_corners(inp_img1, inp_img2, hom, hom_inverse)
	stitched_img=numpy.zeros((stitched_rows, stitched_cols, 3))
	img1_rows, img1_cols, _=inp_img1.shape
	for row in range(0, img1_rows):
		for col in range(0, img1_cols):
			stitched_img[row+row_add][col+col_add]= inp_img1[row][col]
	for row in range(0, stitched_img.shape[0]):
		for col in range(0, stitched_img.shape[1]):
			st_col, st_row=project(col-col_add, row-row_add, hom)
			row_bool = st_row>=0 and st_row<inp_img2.shape[0]
			col_bool = st_col>=0 and st_col<inp_img2.shape[1]
			if row_bool and col_bool:
				stitched_img[row][col] = cv2.getRectSubPix(inp_img2, (1, 1), (st_col, st_row))
				
	
	return stitched_img
	
##get rainer and box outputs in step 3 and 4
def rainerRansac(inlierThresh, iterations):
	image1, image2 = cv2.imread('given_images/Rainier1.png'), cv2.imread('given_images/Rainier2.png')
	mts, insPts1, insPts2 = find_matches(image1, image2)
	inliers_mts, hom, hom_inverse = RANSAC (mts , iterations, inlierThresh, insPts1, insPts2)
	if inliers_mts is None:
		print("Not enough matches")
		return 
	ransacout = cv2.drawMatches(image1, insPts1, image2, insPts2, inliers_mts, None)
	cv2.imwrite("3.png", ransacout)
	print("RANSAC output of Rainer1 and Rainer2 is saved as 3.png")	
	try:
		stitched_img=numpy.uint8(stitch(image1, image2, hom, hom_inverse))
		cv2.imwrite("4.png",stitched_img)
	except:
		print("The images cannot be stitched")
				
	print("Stitched image of Rainer1 and Rainer2 is saved as 4.png")

## to start the program
def start():
	inlierThresh = 5
	iterations = 1000
	
	a2start()
	
	print("\n****STEP3 and STEP4****")	
	rainerRansac(inlierThresh, iterations)
	
	
	
	##Rainer Images
	inp_img = ['given_images/Rainier1.png', 'given_images/Rainier2.png', 'given_images/Rainier3.png', 'given_images/Rainier4.png', 'given_images/Rainier5.png', 'given_images/Rainier6.png']
	
	##MelakwaLake Images
	##inp_img = ['given_images/MelakwaLake1.png', 'given_images/MelakwaLake2.png', 'given_images/MelakwaLake3.png', 'given_images/MelakwaLake4.png']
	
	##Own panoroma 1 Images
	##inp_img = ['road/testimg2.jpg','road/testimg1.jpg', 'road/testimg3.jpg']
	
	##Own panoroma 2 Images
	##inp_img = ['building/building2.jpg','building/building3.jpg', 'building/building1.jpg']
	
	
	
	inp_img = [cv2.imread(img_name) for img_name in inp_img]
	image1 = inp_img[0]
	
	for loop in range(1, len(inp_img)):
		print(loop)
		image2 = inp_img[loop]
		
		mts, insPts1, insPts2 = find_matches(image1, image2)
		matches_img = cv2.drawMatches(image1, insPts1, image2, insPts2, mts, None)
		cv2.imwrite("Extra Output For panoroma\O"+str(loop)+"_Without_Ransac_matches.png",matches_img)
		
		inliers_mts, hom, hom_inverse = RANSAC (mts, iterations, inlierThresh, insPts1, insPts2)
		if inliers_mts is None:
			print("Not enough matches")
			break
		print("Number of inliers_mts: ", len(inliers_mts))
		
		ransacout = cv2.drawMatches(image1, insPts1, image2, insPts2, inliers_mts, None)
		cv2.imwrite("Extra Output For panoroma\O"+str(loop)+"_Ransac_matches.png", ransacout)
		
		try:
			stitched_img=numpy.uint8(stitch(image1, image2, hom, hom_inverse))
			cv2.imwrite("Extra Output For panoroma\O"+str(loop)+"_stitched_img.png",stitched_img)
		except:
			print("The images cannot be stitched")
			break		
		
		image1=stitched_img
		
	cv2.imwrite("panoroma.png",image1)
	print("\npanoroma created as saved as panoroma.png")
	
	aspect_ratio=image1.shape[1]/image1.shape[0]
	cv2.namedWindow("Panoroma", cv2.WINDOW_NORMAL)
	cv2.resizeWindow('Panoroma', int(aspect_ratio*600), 600)
	cv2.imshow("Panoroma",image1)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	cv2.waitKey(1)
	
	
start()
cv2.waitKey(0)