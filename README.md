## Panaroma-Building-Software
This is the assignment of COMP6341 COMPUTER VISION. The aim of this assignment is to build a software to stitch multiple images together to form panoramas. This assignment uses the feature descriptor built in [this](https://github.com/DhwaniSondhi/Feature-Descriptor) repository.

### Description
**FEATURE DETECTION AND MATCHING**<br/>
- This follows the same steps described in [this](https://github.com/DhwaniSondhi/Feature-Descriptor) feature-descriptor.
- Built an additional seperate code using inbuilt functions of [SIFT](https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_sift_intro/py_sift_intro.html) for comparison of output. 

**RANSAC IMPLEMENTATION**<br/>
Computed the homography between the images using RANSAC through following steps:
- **project(col1, row1, hom)**: It helps in projecting the point(x, y) in another image using the homography created.
- **computeInlierCount(H, matches, numMatches, inlierThreshold)**: It helps in finding the number of inlying points given a homography.
- **RANSAC (matches , numMatches, numIterations, inlierThreshold, hom, homInv, image1Display, image2Display)**: It takes the potentially matching points between two images and returns the homography transformation.

For a selected number of iterations, 4 random pairs of potentially matching points are taken. For this, homography is computed which is further used to find the number of inliers. The best homography is selected depending on the highest number of inliers. New inliers are computed for the best homography. These inliers helps in computing the final homography. 

**PANORAMA MOSAIC STITCHING**<br/>
Implemented the function stitch(image1, image2, hom, homInv, stitchedImage):
- Computed the size of the new stiched image by projecting the corners of second image onto the first image. 
- Copied the first image on the stiched image.
- Projected each pixel in stiched image onto second image. 
- Blended the pixel value to the stiched image if the pixel lies within the boundary of second image.

### How to Run the code?
- Set up an environment with python version: 3.5.1 and open-contrib version: 3.3.1.
- Go to “code and input” folder.
- If you want to see the panorama for the dummy images, the simply run the code.py. Else, open code.py, then go to “Start” function select the list of images and uncomment it if required.
- Run the file.
- Final stitched image is displayed and saved as panoroma.png.

### Outputs
<img src="https://github.com/DhwaniSondhi/Panaroma-Building-Software/blob/master/results/Rainer%20panorama.png" alt="alt" width="700" height="400"/> <br/>
<img src="https://github.com/DhwaniSondhi/Panaroma-Building-Software/blob/master/results/building%20panorama.png" alt="alt" width="700" height="400"/> <br/>
<img src="https://github.com/DhwaniSondhi/Panaroma-Building-Software/blob/master/results/MelakwaLake%20panorama.png" alt="alt" width="700" height="400"/> <br/>

[Click here for more outputs](https://github.com/DhwaniSondhi/Panaroma-Building-Software/tree/master/results) <br/>
[Click here for more description of the code and outputs](https://github.com/DhwaniSondhi/Panaroma-Building-Software/blob/master/Report.pdf) <br/>
[Click here for the assignment description](https://github.com/DhwaniSondhi/Panaroma-Building-Software/blob/master/Project%20Description.pdf)

