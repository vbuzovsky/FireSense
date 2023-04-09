import numpy as np
import cv2
import copy

# private function for 'draw_optical_flow'
def _draw_flow(img, flow, step=16):

    h, w = img.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1).astype(int)
    fx, fy = flow[y,x].T

    lines = np.vstack([x, y, x-fx, y-fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)

    img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.polylines(img_bgr, lines, 0, (0, 255, 0))

    for (x1, y1), (_x2, _y2) in lines:
        cv2.circle(img_bgr, (x1, y1), 1, (0, 255, 0), -1)

    return img_bgr

# private function for 'draw_optical_flow'
def _draw_hsv(flow):

    h, w = flow.shape[:2]
    fx, fy = flow[:,:,0], flow[:,:,1]

    ang = np.arctan2(fy, fx) + np.pi
    v = np.sqrt(fx*fx+fy*fy)

    hsv = np.zeros((h, w, 3), np.uint8)
    hsv[...,0] = ang*(180/np.pi/2)
    hsv[...,1] = 255
    hsv[...,2] = np.minimum(v*4, 255)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return bgr


# loops through list of images, calculate average / sum of Farneback flow
def calculate_optical_flow(imgs_lst : list):
    grayscale_imgs_lst = []
    optical_flows = []

    # Convert all frames to grayscale
    for img in imgs_lst:
        grayscale_imgs_lst.append(cv2.cvtColor(img[0], cv2.COLOR_BGR2GRAY))

    # iterate over grayscale images, take consecutive frames and calculate flow betweem them
    # generates 9 flows between 10 frames (1 x 2 x 3 x 4 x 5 x 6 x 7 x 8 x 9 x 10); x = flow calc between frames
    for i in range(len(grayscale_imgs_lst)):
        if not(i+1 == len(grayscale_imgs_lst)):
            # uncomment to see what grayscale pairs of frames are being used:
            #print(f'currennt {i}: ', grayscale_imgs_lst[i])
            #print(f'next {i+1}: ', grayscale_imgs_lst[i+1])
            optical_flows.append(cv2.calcOpticalFlowFarneback(grayscale_imgs_lst[i], grayscale_imgs_lst[i+1], None, 0.5, 3, 15, 3, 5, 1.2, 0))

    # create deep copy of one of the flows | this gets replaced by sum of all flows, need this to have same data structure
    optical_flow_sum = copy.deepcopy(optical_flows[0])

    # implementation of naive sum over all calculated optical flows
    for i, flow in enumerate(optical_flows):
        if not(i+1 == len(optical_flows)):
            for x, outer in enumerate(flow):
                for y, inner in enumerate(outer):
                    optical_flow_sum[x][y] += optical_flows[i][x][y] + optical_flows[i+1][x][y]
    
    return optical_flow_sum

# param = 'flow' / 'hsv' print
def draw_optical_flow(flow, frame, param : str = 'flow'):
    if(param == 'flow'): # Draw flow
        grayscale_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        while True:
            cv2.imshow("OPTICAL FLOW", _draw_flow(grayscale_frame, flow))
            key = cv2.waitKey(5)
            if key == ord('q'):
                break
    else: # draw hsv
        while True:
            cv2.imshow('HSV FLOW', _draw_hsv(flow))
            key = cv2.waitKey(5)
            if key == ord('q'):
                break
        



