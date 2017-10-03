# Advanced Lane Line Finding
In this project, your goal is to write a software pipeline to identify the lane boundaries in a video, but the main output or product we want you to create is a detailed writeup of the project.
The project has the following steps:
* Camera calibration
* Distortion correction
* Color and gradient threshold
* Perspective Transform
* Finding lane lines
* Calculate the curvature and the position of the vehicle
* Video processing

[Source File](Advanced-Lane-Lines.ipynb)
## Step 1 Camera calibration
The first step of the image processing pipeline is camera calibration in order to fix the image distortation problem from the effect of camera lens. The project provides 20 chessboard camera images to help us do the camera calibration. I use the cv2.findChessboardCorners() function in order to find all chessboard corners on these images. After that, I use the cv2.calibrateCamera() function to find the distortion array of the camera lens. Finally, I apply the cv2.undistort() function to get undistorted images.
The output of this step is saved on [output_images/camera_cal folder](output_images/camera_cal)
```python
# Find the camera image calibration points based on the chaessboard images from camera
def find_Calibration_Points(image_paths, nx=9, ny=6):
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((nx*ny,3), np.float32)
    objp[:,:2] = np.mgrid[0:nx, 0:ny].T.reshape(-1,2)
    objpoints = [] # 3d points in real world space
    imgpoints = [] # 2d points in image plane
    for idx, image_path in enumerate(image_paths):
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)
    return objpoints, imgpoints
# load test images
cali_images = glob.glob('camera_cal/*.jpg')
objpoints, imgpoints = find_Calibration_Points(cali_images)

# Do camera calibration given object points and image points
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size,None,None)


# Save the camera calibration result for later use (we won't worry about rvecs / tvecs)
dist_pickle = {}
dist_pickle["mtx"] = mtx
dist_pickle["dist"] = dist
pickle.dump( dist_pickle, open( "camera_cal/wide_dist_pickle.p", "wb" ) )
```
## Step 2 Distortion correction
After do the camera calibration, I have the camera distortion arrays. Now I apply it to undistort the test images and save the undistorted images to [output folder](output_images/test_images/undistorted).
```python
TEST_IMAGES_FOLDER = 'test_images'
# Test undistort on test_images
test_image_paths = glob.glob(TEST_IMAGES_FOLDER + '/*.jpg')
test_images, test_gray_images = load_images(test_image_paths)
test_dst_images = []
for idx, img in enumerate(test_images):
    dst = cv2.undistort(img, mtx, dist, None, mtx)
    test_dst_images.append(dst)
    if debug:
        # Visualize undistortion
        show2images(img, idx + 1, dst, 'Undistorted image')
        cv2.imwrite('{}/{}/{}/{}.jpg'.format(OUTPUT_IMAGES, TEST_IMAGES_FOLDER, 'undistorted',idx + 1), dst)
```
## Step 3 Color and gradient threshold
I follow the Udacity lession and try many color and threshold approaches in order to choose the best result. Please reference to my [source code](Advanced-Lane-Lines.ipynb) on Step 3 Color and gradient threshold for more detail. Finally, I choose the combination of the sobel threshold x direction and the S (saturation) channel color threshold approach and get the expected output on [output_images/test_images/threshold](output_images/test_images/threshold) folder. Below is more detail on my approach.
### Sobel threshold x direction
I apply the Sobel operator in order to take the derivative of the image in the x or y direction. I try with the sobel threshold operation in the x direction on the test images. The output shows that it can work well with solid lines but not on dashed lines, and also cannot detect lines on the bright condition (image 3, 6 and 7).
```python
# Apply the sobel operator function to take the derivative of the image in the x or y orient.
def abs_sobel_thresh(gray, orient='x', thresh_min=0, thresh_max=255):
    
    if orient == 'x':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    else:
        sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
    
    abs_sobel = np.absolute(sobel)
    
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    
    sxbinary = np.zeros_like(scaled_sobel)
    
    sxbinary[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1
    
    return sxbinary

for idx, dst in enumerate(test_dst_images):
    gray = cv2.cvtColor(dst, cv2.COLOR_RGB2GRAY)
    abs_sobel_img = abs_sobel_thresh(gray, orient='x', thresh_min=30, thresh_max=100)
    if debug:
        show3images(dst, idx + 1, gray, 'Gray', abs_sobel_img, 'Sobel X image')
```
### S (saturation) color channel threshold
It's easy recognize that the lighting information does not have so much value for lane detection and it makes a lot of noise on output. So I try to convert the image from RGB to HLS (hue, lightness, and saturation) color space and apply color thresholds on S (saturation) and H (hue) channels.
Applying threshold on the S (saturation) channel give the output that near my expectaction. I can detect lines under bright condition however it misses some dashed lines.
```python
# The function to apply threshold to a single channel image.
def channel_select(c_image, thresh=(0,255)):
    binary_output = np.zeros_like(c_image)
    binary_output[(c_image > thresh[0]) & (c_image <= thresh[1])] = 1
    return binary_output
# Try to apply threshold on the S channel image
for idx, dst in enumerate(test_dst_images):
    hls = cv2.cvtColor(dst, cv2.COLOR_RGB2HLS)
    S = hls[:,:,2]
    s_binary = channel_select(S, thresh=(150, 255))
    if debug:
        show3images(dst, idx + 1, S, "S Channel", s_binary, "S Binary")
print('Done')
```
### Combine sobel x and S color threshold
From the abobe results, I have realized that I can combine the output of the sobel threahold x direction and the S (saturation) channel color threshold together in order to get the expected result. The lines are detected well on both dashed and solid lines and it also work well on the bright condition.
```python
def combine1(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    #gradx = abs_sobel_thresh(gray, orient='x', thresh_min=30, thresh_max=100)
    gradx = abs_sobel_thresh(gray, orient='x', thresh_min=60, thresh_max=255)
    
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    S = hls[:,:,2]
    s_binary = channel_select(S, thresh=(150, 255))
    #s_binary = channel_select(S, thresh=(120, 255))
    
    combined = np.zeros_like(s_binary)
    #combined[((gradx == 1) & (grady == 1)) & ((mag_binary == 1) & (dir_binary == 1))] = 1
    #combined[((gradx == 1) & (dir_binary == 1)) | ((mag_binary == 1) & (grady == 1))] = 1
    combined[((gradx == 1) | (s_binary == 1))] = 1
    return combined
for idx, dst in enumerate(test_dst_images):
    combined = combine1(dst)
    #combined[((gradx == 1) | ((dir_binary == 1) & (mag_binary == 1)))] = 1
    if debug:
        show3images(dst, idx + 1, gray, 'Gray', combined, 'Combined image')
print('Done')
```
### Region of Interest
Following the project 1, I also crop the input image in order to filter out some noises such as skyline, other cars, trees, etc...
```python
def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


# define the region of interest vertices based on image size
def calculate_roi(img):
    rows, cols = img.shape[:2]
    # 1280 - 720
    p1 = [cols*0.1, rows*0.95]
    p2 = [cols*0.4, rows*0.6]
    p3 = [cols*0.6, rows*0.6] 
    p4 = [cols*0.9, rows*0.95]
    vertices = np.array([[p1, p2, p3, p4]], dtype=np.int32)
    return vertices

for idx, dst in enumerate(test_dst_images):
    combined1 = combine1(dst)
    roi_img = region_of_interest(combined1, calculate_roi(combined1))
    if debug:
        show3images(dst, idx + 1, combined1, 'Combine 1', roi_img, 'ROI')
```
## Step 4 Perspective Transform
The purpose of this step is transforming the image to the birds eye view in oder to fit a curve on the lane easier. I manually choose the 4 source points that similiar to the region of masking. And I choose the 4 destination points that will transform the source points to the birds view eye. I use the cv2.getPerspectiveTransform() to get the transformation matrix and use it for transform image by the cv2.warpPerspective() function. I also get the revert transformation matrix to revert the warped image to the original image after fit the lane line poly. The output of this step is saved to [output_images/test_images/warped](output_images/test_images/warped) folder.
```python
def warp(img):
    #img_size = (img.shape[1], img.shape[0])
    #src = np.float32([[595, 450],[695,450],[1045,675],[260,680]])
    #dst = np.float32([[335,95],[1120,95],[1045,675],[260,680]])
    img_size = (img.shape[0], img.shape[1])
    offset = 210
    img_size = (img.shape[1], img.shape[0])
    src = np.float32([[150 + 430, 460], [1150 - 440, 460], [1150, 720], [150, 720]])
    dst = np.float32([[offset, 0], [img_size[0]-offset, 0], [img_size[0]-offset, img_size[1]], [offset, img_size[1]]])
    #img_size = (img.shape[1], 223)
    #src = np.float32([[0, 673], [1207, 673], [0, 450], [1280, 450]])
    #dst = np.float32([[569, 223], [711, 223], [0, 0], [1280, 0]])
    
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    warped = cv2.warpPerspective(img, M, img_size)
    
    return warped, M, Minv
for idx, dst in enumerate(test_dst_images):
    roi_img = region_of_interest(dst, calculate_roi(dst))
    warped, M, Minv = warp(roi_img)
    if debug:
        show_warped(roi_img, warped)
def pipeline(img):
    dst = cv2.undistort(img, mtx, dist, None, mtx)
    combined1 = combine1(dst)
    roi_img = region_of_interest(combined1, calculate_roi(combined1))
    return roi_img
for idx, img in enumerate(test_images):
    pine_img = pipeline(img)
    warped, M, Minv = warp(pine_img)
    if debug:
        show3images(dst, idx + 1, pine_img, 'Pipeline', warped, 'Warped')      
```
