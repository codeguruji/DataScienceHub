#video stabilization
import cv2
import os
import numpy as np
from matplotlib import pyplot as plt

os.chdir(r"C:\Users\schoud790\Documents\Python Codes\computer vision")
video_file = "Traffic - 27260.mp4"

vid_cap = cv2.VideoCapture(video_file) #obtain the video capture from the input video file

#get video stream properties
n_frames = np.int(vid_cap.get(cv2.CAP_PROP_FRAME_COUNT))
frame_w = np.int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_h = np.int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = 25

#define the video codecs for the output video
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
output_vid = cv2.VideoWriter("output.mp4", fourcc, fps, (frame_w, frame_h))

# Pre-define transformation-store array
transforms = np.zeros((n_frames-1, 3), np.float32)

#read the first frame
_, prev = vid_cap.read()
prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

for i in range(n_frames-2):
    # Detect feature points in previous frame
    prev_pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=200,\
                                       qualityLevel=0.01, minDistance=30,\
                                       blockSize=3)                                                                        
    # Read next frame
    success, curr = vid_cap.read() 
    if not success:
        break
    # Convert to gray
    curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)

    # Calculate optical flow (i.e. track feature points) (Lucas-Kanade algo)
    # Status gives if the current frame contains the points from the prev frame
    curr_pts, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None)

    # Sanity check
    assert prev_pts.shape == curr_pts.shape

    # Filter only valid points
    idx = np.where(status==1)[0]
    prev_pts = prev_pts[idx]
    curr_pts = curr_pts[idx]

    # Find transformation matrix
    # m = cv2.estimateRigidTransform(prev_pts, curr_pts, fullAffine=False) #for opencv 3 or less
    m, inliers = cv2.estimateAffinePartial2D(prev_pts, curr_pts)

    ## m = [[cos(x), -sin(x), t(x)], [sin(x), cos(x), t(y)]]

    # extract translation
    dx = m[0,2]
    dy = m[1,2]

    # Extract rotation angle
    da = np.arctan2(m[1,0], m[0,0])

    # Store transformation
    transforms[i] = [dx,dy,da]

    #move to next frame
    prev_gray = curr_gray

    for p in prev_pts:
        p = p[0]
        prev = cv2.circle(prev, (p[0], p[1]), radius=5, color=(100,0,0), thickness=3)
    cv2.imshow("a", prev)
    cv2.waitKey(1)
    prev = curr

    if i % 20 == 0:
        print("Frame: " + str(i+1) +  "/" + str(n_frames) + " -  Tracked points : " + str(len(prev_pts)))

cv2.destroyAllWindows()
# to find a smooth motion we need to find the cummulative differential motions
# Calculate smooth motion between frames
# Compute trajectory using cumulative sum of transformations
trajectory = np.cumsum(transforms, axis=0)

#plt.plot(range(ma.shape[0]), ma[:,2])
#plt.show()

# Smoothen the motion using a moving average filter
def movingAverage(curve, radius):
    window_size = 2 * radius + 1
    # Define the filter 
    f = np.ones(window_size)/window_size 
    # Add padding to the boundaries 
    curve_pad = np.lib.pad(curve, (radius, radius), 'edge') 
    # Apply convolution 
    curve_smoothed = np.convolve(curve_pad, f, mode='same') 
    # Remove padding 
    curve_smoothed = curve_smoothed[radius:-radius]
    # return smoothed curve
    return curve_smoothed

smoothed = np.copy(trajectory)
for i in range(3):
    smoothed[:,i] = movingAverage(smoothed[:,i], 2)

# Calculate difference in smoothed_trajectory and trajectory
difference = smoothed - trajectory

# Calculate newer transformation array
transforms_smooth = transforms + difference

## Apply the smoothed camera motion to the frames
# Reset stream to first frame 
vid_cap.set(cv2.CAP_PROP_POS_FRAMES, 0) 
 
# Write n_frames-1 transformed frames
for i in range(n_frames-2):
    # Read next frame
    success, frame = vid_cap.read() 
    if not success:
        break
    
    # Extract transformations from the new transformation array
    dx = transforms_smooth[i,0]
    dy = transforms_smooth[i,1]
    da = transforms_smooth[i,2]
    
    #Reconstruct transformation matrix accordingly to new values
    m = np.zeros((2,3), np.float32)
    m[0,0] = np.cos(da)
    m[0,1] = -np.sin(da)
    m[1,0] = np.sin(da)
    m[1,1] = np.cos(da)
    m[0,2] = dx
    m[1,2] = dy

    # Apply affine wrapping to the given frame
    frame_stabilized = cv2.warpAffine(frame, m, (frame_w,frame_h))

    # Fix border artifacts
    frame_stabilized = fixBorder(frame_stabilized)

    # Write the frame to the file
    frame_out = cv2.hconcat([frame, frame_stabilized])
    
    # If the image is too big, resize it.
    if(frame_out.shape[1] > 1920):
        frame_out = cv2.resize(frame_out, (frame_out.shape[1]//2, frame_out.shape[0]//2))
  
    #cv2.imshow("Before and After", frame_out)
    #cv2.waitKey(10)
    output_vid.write(frame_stabilized)

output_vid.release()


def fixBorder(frame):
    s = frame.shape
    # Scale the image 4% without moving the center
    T = cv2.getRotationMatrix2D((s[1]/2, s[0]/2), 0, 1.04)
    frame = cv2.warpAffine(frame, T, (s[1], s[0]))
    return frame

a = np.convolve(curve[:,0], f, mode='same')s
a.shape
    # Apply convolution
    curve_smoothed = np.convolve(curve_pad, f, mode='same')
    ma = np.zeros(curve.shape)
    for i in range(radius//2):
        ma[i] = curve[i]
        ma[-(i+1)] = curve[-(i+1)]

    for i in range(radius//2, len(curve)-radius//2):
        ma[i] = np.sum(curve[i-(radius//2):i+(radius//2)], axis=0)/radius
    return ma

def display(img):
    cv2.imshow("img", img)
    cv2.waitKey()

#get good features to track between frames
goodFeaturesToTrack

#detect optical flow (Lucas-kanade Optical Flow)
calcOpticalFlowPyrLK

#estimate motion
estimateRigidTransform

#motion decomposition
