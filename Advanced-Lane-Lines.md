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

