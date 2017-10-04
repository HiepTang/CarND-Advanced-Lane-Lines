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

Output:
* [Project video output](project_video_outout.mp4)
* [Challenge video output](challenge_video_outout.mp4)
* [Camera calibration output](output_images/camera_cal)
* [Distortion correction output](output_images/test_images/undistorted)
* [Color and gradient threshold output](output_images/test_images/threshold)
* [Perspective Transform output](output_images/test_images/warped)
* [Finding lane lines output](output_images/test_images/lane)
* [Final processing test images output](output_images/test_images/final)

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
## Step 5 Finding lane lines
The purpose of this step is finding the lane lines on the warped image. The output of this step is saved on [output_images/test_images/lane](output_images/test_images/lane) folder.
### Define the Line and Lane classes
The Line class use to store the properties of each line detection such as the x, y points, the curvature and the line fit.
```python
# Define a class to receive the characteristics of each line detection
class Line():
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False  
        # x values of the last n fits of the line
        self.recent_xfitted = [] 
        #average x values of the fitted line over the last n iterations
        self.bestx = None     
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None  
        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]  
        #radius of curvature of the line in some units
        self.radius_of_curvature = None 
        #distance in meters of vehicle center from the line
        self.line_base_pos = None 
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float') 
        #x values for detected line pixels
        self.allx = None  
        #y values for detected line pixels
        self.ally = None
```
The Lane class stores the lane detection properties: left line and right line.
```python
# Define a class for each Lane detection
class Lane():
    def __init__(self):
        # was the lane detected in the last iteration?
        self.detected = False
        # the left line
        self.left_line = None
        # the right line
        self.right_line = None
```
### Finding lane lines using the sliding windows algorithm
Based on the histogram of the warped image, I can find the left and right x base points. From these points, I use the sliding windows to move up in order to find the line pixels within the sliding window.
```python
# nwindows - the number of sliding windows
# margin - the margin width of the windows
# minpix - the minimun pixels found to recenter window
def find_lanes(warped, nwindows=9, margin=100, minpix=50):
    # Take a histogram of the bottom half of the image
    histogram = np.sum(warped[warped.shape[0]//2:,:], axis=0)
    
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    
    # Set height of sliding windows
    window_height = np.int(warped.shape[0]/nwindows)
    
    # Identity the x and y positions of all nonzero pixels in the image
    nonzero = warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []
    
    # Step through the windows on by one
    for window in range(nwindows):
        # Identify the window boundaries in x and y (and left and right)
        win_y_low = warped.shape[0] - (window+1)*window_height
        win_y_high = win_y_low + window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        
        # Identify the nonzero pixels in x and y within the window
        #good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        #(nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high)
                         & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high)
                          & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        
        # if you found > minpix pixels, recenter next window on their mean position
        if (len(good_left_inds) > minpix):
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if (len(good_right_inds) > minpix):
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
        
    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)
    
    # Extract left and right lane pixel positions
    leftx =  nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    
    left_line = Line()
    
    right_line = Line()
    
    if ((leftx.size > 0) and (lefty.size > 0)):
    # Fit a second order polynomiral to each
        left_fit = np.polyfit(lefty, leftx, 2)
        left_line.detected = True
    else:
        left_fit = None
        left_line.detected = False
    if ((rightx.size > 0) and (righty.size > 0)):
        right_fit = np.polyfit(righty, rightx, 2)
        right_line.detected = True
    else:
        right_fit = None
        right_line.detected = False
    
    left_line.allx = leftx
    left_line.ally = lefty
    left_line.current_fit = left_fit
    
    right_line.allx = rightx
    right_line.ally = righty
    right_line.current_fit = right_fit
    
    
    lane = Lane()
    lane.detected = (left_line.detected and right_line.detected)
    lane.left_line = left_line
    lane.right_line = right_line
    return lane
```
I try to test the find_lanes() function with the test images and the result is expected.
```python
# Visual the found lane lines for testing
def visual_lanes(warped):
    
    lane = find_lanes(warped, nwindows=9, margin=100, minpix=50)
    left_line = lane.left_line
    right_line = lane.right_line
    
    left_fit = left_line.current_fit
    right_fit = right_line.current_fit
    # Generate x and y values for plotting
    ploty = np.linspace(0, warped.shape[0]-1, warped.shape[0])
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    
    out_img = np.dstack((warped, warped, warped)) * 255
    
    out_img[left_line.ally, left_line.allx] = [255,0,0]
    out_img[right_line.ally, right_line.allx] = [0,0,255]
 
    return out_img, left_fitx, right_fitx
 for idx, img in enumerate(test_images):
    pine_img = pipeline(img)
    warped, M, Minv = warp(pine_img)
    out_img, left_fitx, right_fitx = visual_lanes(warped)
    if debug:
        show_lane_images(img, idx + 1, warped, 'Warped', out_img, 'Lane image', left_fitx, right_fitx)
```
### Finding lane lines from the previous frame lane
Finding lane with the sliding windows algorithm can work well. However, it cost time to execute. It's easy to regconize that the lanes are not much different between 2 next frames. So, I implement the find lane from previous lane function following the Udacity code on lession.
```python
def find_lanes_from_previous(warped, lane, margin=100):
    nonzero = warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    
    left_fit = lane.left_line.current_fit
    right_fit = lane.right_line.current_fit
    
    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + 
    left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + 
    left_fit[1]*nonzeroy + left_fit[2] + margin))) 

    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + 
    right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + 
    right_fit[1]*nonzeroy + right_fit[2] + margin)))
    
    # Extract left and right line pixels positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    
    left_line = Line()
    right_line = Line()
    
    # Fit the second order polynomial to each
    if (leftx.size > 0 and lefty.size > 0):
        left_fit = np.polyfit(lefty, leftx, 2)
        left_line.detected = True
    else:
        left_fit = None
        left_line.detected = False
        if debug:
            print('Cannot detect left line')
    if (rightx.size > 0 and righty.size > 0):
        right_fit = np.polyfit(righty, rightx, 2)
        right_line.detected = True
    else:
        right_fit = None
        right_line.detected = False
        if debug:
            print('Cannot detect right line')
    
    
    
    left_line.allx = leftx
    left_line.ally = lefty
    left_line.current_fit = left_fit
    
    right_line.allx = rightx
    right_line.ally = righty
    right_line.current_fit = right_fit
    
    current_lane = Lane()
    current_lane.detected = (left_line.detected and right_line.detected)
    current_lane.left_line = left_line
    current_lane.right_line = right_line
    return current_lane
```
## Step 6 Calculate the curvature and the position of the vehicle
Following [the avesome tutorial](http://www.intmath.com/applications-differentiation/8-radius-curvature.php) and the Udacity lession, I can implement a function to calculate the radius of curvature of the lane. I also assume that the lane is about 30 meters long and 3.7 meters wide in order to convert the radius of curvature in pixels to meters.
I assume that the car always on the center of the lane, so I can calculate the car position by the average of left and right x bottom points. I convert the car position pixel value to meter with the same 3.7 meters wide assumption.
```python
# ym_per_pix - meters per pixel in y dimension
# xm_per_pix - meters per pixel in x dimension
def cal_curvature(warped, lane, ym_per_pix = 30/720, xm_per_pix = 3.7/800):
    ploty = np.linspace(0, warped.shape[0]-1, warped.shape[0])
    y_eval = np.max(ploty)
    leftx = lane.left_line.allx
    lefty = lane.left_line.ally
    
    rightx = lane.right_line.allx
    righty = lane.right_line.ally
    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)
    left_curverad, right_curverad = (0,0)
    # Calculate the new radii of curvature
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    return left_curverad, right_curverad

def cal_distance(warped, lane, xm_per_pix = 3.7/800):
    h = warped.shape[0]
    left_fit_x_int = lane.left_line.current_fit[0]*h**2 + lane.left_line.current_fit[1]*h + lane.left_line.current_fit[2]
    right_fit_x_int = lane.right_line.current_fit[0]*h**2 + lane.right_line.current_fit[1]*h + lane.right_line.current_fit[2]
    lane_center_position = (left_fit_x_int + right_fit_x_int) / 2
    
    car_position = warped.shape[1]/2
    center_dist = (car_position - lane_center_position) * xm_per_pix
    return center_dist

for idx, img in enumerate(test_images):
    pine_img = pipeline(img)
    warped, M, Minv = warp(pine_img)
    lane = find_lanes(warped, nwindows=9, margin=100, minpix=50)
    left_curverad, right_curverad = cal_curvature(warped, lane)
    lane.left_line.radius_of_curvature = left_curverad
    lane.right_line.radius_of_curvature = right_curverad
    center_dist = cal_distance(warped, lane)
    print(idx + 1, left_curverad, right_curverad, center_dist)
```
## Putting together - the image processing pipeline
### Define the draw lane function
The purpose of this function is to draw the found lane into the original image. I use the inverted transformation matrix from the perspective transformation step in order to transform to the original image.
```python
# The draw lane function: Draw the found lane into the original image
# Minv: The inverted transformation matrix that return from the warped function.
def draw_Lane(warped, lane, Minv, image):
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    
    ploty = np.linspace(0, warped.shape[0]-1, warped.shape[0] )
    left_fit = lane.left_line.current_fit
    right_fit = lane.right_line.current_fit
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    
    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (image.shape[1], image.shape[0])) 
    # Combine the result with the original image
    result = cv2.addWeighted(image, 1, newwarp, 0.3, 0)
    return result

# Draw the curvature and distance into the original image
def draw_curv_dist(img, curv, dist):
    text = 'Curvature radius: ' + '{:04.2f}'.format(curv) + 'm'
    cv2.putText(img, text, (40,70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (200,255,155), 2, cv2.LINE_AA)
    
    direction = ''
    if dist > 0:
        direction = 'right'
    elif dist < 0:
        direction = 'left'
    abs_dist = abs(dist)
    text = '{:04.3f}'.format(abs_dist) + 'm ' + direction + ' of center'
    cv2.putText(img, text, (40,120), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (200,255,155), 2, cv2.LINE_AA)
    return img
```
### Test the pipeline on the test images
I implement the pipeline in order to test the images from test_images folder. The output is saved to [output_images/test_images/final](output_images/test_images/final) folder.
```python
for idx, img in enumerate(test_images):
    pine_img = pipeline(img)
    warped, M, Minv = warp(pine_img)
    lane = find_lanes(warped, nwindows=9, margin=100, minpix=50)
    left_curverad, right_curverad = cal_curvature(warped, lane)
    center_dist = cal_distance(warped, lane)
    curvered = (left_curverad + right_curverad) / 2
    result = draw_Lane(warped, lane, Minv, img)
    final = draw_curv_dist(result, curvered, center_dist)
    if debug:
        show3images(img, idx + 1, pine_img, 'Pipeline', final, 'Result')
```
## Step 7 Video Processing
### Implement the sanity check function
In order to evaluate the found lane is good enough, I implement the sanity check function with do the following checks: check the distance between 2 left and right lines and check these two lines are roughly parallel.
```python
import math
# Test distance from left line and right line
# The distance horizontally between left line and right line should be about 800 +- 100 pixels
def test_distance(warped, lane):
    h = warped.shape[0]
    left_fit_x_int = lane.left_line.current_fit[0]*h**2 + lane.left_line.current_fit[1]*h + lane.left_line.current_fit[2]
    right_fit_x_int = lane.right_line.current_fit[0]*h**2 + lane.right_line.current_fit[1]*h + lane.right_line.current_fit[2]
    
    x_int_diff = abs(right_fit_x_int - left_fit_x_int)
    if (abs(x_int_diff - 800) > 100):
        if debug:
            print('Test distance failure: ', x_int_diff)
        return False
    else:
        return True
def test_with_previous_lane(warped, previous_lane, lane):
    h = warped.shape[0]
    delta_left = math.fabs(lane.left_line.allx[h-1] - previous_lane.left_line.allx[h-1])
    if delta_left > 5:
        if debug:
            print('delta left failure', delta_left)
        return False
    delta_right = math.fabs(lane.right_line.allx[h-1] - previous_lane.right_line.allx[h-1])
    if delta_right > 5:
        return False
    return True

# Test to check the left line and right line are roughly parallel
def test_parallel(warped, lane):
    top = 4
    bottom = warped.shape[0]
    middle = bottom//2
    
    left_fit = lane.left_line.current_fit
    right_fit = lane.right_line.current_fit
    left_fit_x_bottom_int = left_fit[0]*bottom**2 + left_fit[1]*bottom + left_fit[2]
    right_fit_x_bottom_int = right_fit[0]*bottom**2 + right_fit[1]*bottom + right_fit[2]
    
    left_fit_x_top_int = left_fit[0]*top**2 + left_fit[1]*top + left_fit[2]
    right_fit_x_top_int = right_fit[0]*top**2 + right_fit[1]*top + right_fit[2]
    
    left_fit_x_middle_int = left_fit[0]*middle**2 + left_fit[1]*middle + left_fit[2]
    right_fit_x_middle_int = right_fit[0]*middle**2 + right_fit[1]*middle + right_fit[2]
    
    x_int_bottom_diff = abs(right_fit_x_bottom_int - left_fit_x_bottom_int)
    x_int_top_diff = abs(right_fit_x_top_int - left_fit_x_top_int)
    x_int_middle_diff = abs(right_fit_x_middle_int - left_fit_x_middle_int)
    
    
    if (abs(x_int_bottom_diff - x_int_top_diff) > 100) or (abs(x_int_top_diff - x_int_middle_diff) > 100) or (abs(x_int_bottom_diff - x_int_middle_diff) > 100):
        if debug:
            print('Test parallel failure: ', x_int_bottom_diff, x_int_middle_diff, x_int_top_diff)
        return False
    else:
        if debug:
            print('Test parallel passed: ', x_int_bottom_diff, x_int_middle_diff, x_int_top_diff)
        return True
# Checked whether the found lane is good enough
def sanity_check(warped, previous_lane, lane):
    if (lane.detected == False):
        return False
    # Check the lane distance is enough
    if (test_distance(warped, lane) == False):
        return False
    # Check two left and right line are roughly parallel
    if (test_parallel(warped, lane) == False):
        return False
    
    return True
    
if debug:
    for idx, img in enumerate(test_images):
        pine_img = pipeline(img)
        warped, M, Minv = warp(pine_img)
        lane = find_lanes(warped, nwindows=9, margin=100, minpix=50)
        print(idx + 1)
        sanity_check(warped, lane, lane)
```
### The process image function
The process image function process one image from video following the below steps:
* Finding lanes using the sliding windows algorithm for the first frame.
* Finding lanes from the previous found lane for the next frames.
* Do sanity check for the found lane.
* If the sanity check is passed, the lane is good enough and draw to the original image.
* If the sanity check is failed and the number of failure is below the MAXIMUM_FAILURE allowed, the previous lane is used and draw to the original image.
* If the sanity check is failed and the number of failure is above or equal the MAXIMUM_FAILURE allowed, the lane will be found again using the sliding window algorithrm.
```python
# Process one image
def process_image(img):
    MAXIMUM_FAILURE = 2
    global last_lane
    global no_of_failure
    pine_img = pipeline(img)
    warped, M, Minv = warp(pine_img)
    # It's the first frame - find lane using the sliding windows
    if (last_lane is None):
        if debug:
            print('Find the first frame')
        lane = find_lanes(warped, nwindows=9, margin=100, minpix=50)
    else:
        if debug:
            print('Find from previous frame')
        lane = find_lanes_from_previous(warped, last_lane, margin=100)
    # Go to sanity check
    if (last_lane is None):
        last_lane = lane
    else:
        # Do the sanity check
        if (sanity_check(warped, last_lane, lane) == False):
            no_of_failure += 1
            if debug:
                print('Sanity check failure: ', no_of_failure)
            # Reset - finding again by sliding window if no of failure is over maximum failure allowed.
            if (no_of_failure >= MAXIMUM_FAILURE):
                if debug:
                    print('Reset - find by siding windows')
                lane = find_lanes(warped, nwindows=9, margin=100, minpix=50)
                # Do sanity check again
                if (sanity_check(warped, last_lane, lane) == False):
                    # Use the previous lane if the sanity check still failure
                    lane = last_lane
                    no_of_failure += 1
                else:
                    # Reset the last lane and no of failure
                    last_lane = lane
                    no_of_failure = 0
            else:
                lane = last_lane
        else:
            last_lane = lane
            no_of_failure = 0
    
    # Calculate the curvature of lane  
    left_curverad, right_curverad = cal_curvature(warped, lane)
    # Calculate the car position
    center_dist = cal_distance(warped, lane)
    curvered = (left_curverad + right_curverad) / 2
    # Draw the found lane
    result = draw_Lane(warped, lane, Minv, img)
    # Draw the curvature and position
    final = draw_curv_dist(result, curvered, center_dist)
    return final
```
### Process the project video
I apply the process image function in order to process the project video. The video output works well with clear detected lane lines and the curvature and position information.

[The project video output](project_video_outout.mp4)

```python
video_input1 = VideoFileClip('project_video.mp4')
video_output = "project_video_outout.mp4"
processed_video = video_input1.fl_image(lambda img:process_image(img))
%time processed_video.write_videofile(video_output, audio=False)
```
### Process the challenge video
I also try to process the challenge video. However, the result is not good with a lot of lane lines that are not detected correctly.

[The challenge video output](challenge_video_outout.mp4)

## Conclusion
My solution can work well on the project video output but not good on the challenge video output. I think the root cause is my color and threshold solution is not good enough in order to get the correct lane lines edge and filter the noise from the light condition. I will improve it in the future.
