import cv2
import numpy as np

import matplotlib.pyplot as plt

def filter_by_essential(kp1, kp2, focus, center):
    E, mask = cv2.findEssentialMat(kp1, kp2, focus, center, cv2.RANSAC, 0.95, 3.0)
    mask = mask.reshape(-1)
    kp1 = kp1[mask==1]
    kp2 = kp2[mask==1]
    return kp1, kp2

def video_cutter(video_path, save_folder, parallax_threshold = 30, b_show=True):
    videoCapture = cv2.VideoCapture(video_path)
    fps = 30 * 2;

    mask_radius = 50;

    # params for ShiTomasi corner detection
    feature_params = dict( maxCorners = 100,
                           qualityLevel = 0.3,
                           minDistance = 15,
                           blockSize = 15 )

    # Parameters for lucas kanade optical flow
    lk_params = dict( winSize  = (15,15),
                      maxLevel = 2,
                      criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # Create some random colors
    # color = np.random.randint(0,255,(100,3))
    # Take first frame and find corners in it
    ret, old_frame = videoCapture.read()
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)

    feature_params['qualityLevel'] = 0.2
    #cv2.imwrite(save_folder+"/"+str(0)+'.jpg', old_frame)

    # Create a mask image for drawing purposes
    # mask = np.zeros_like(old_frame)
    image_id = 1;
    para_interval = 20
    interval = int(parallax_threshold/para_interval)
    while True:
        success, frame = videoCapture.read()
        #success, frame = videoCapture.read()
        if(not success):
            break;
        frame_save = frame.copy()
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # calculate optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
        # Select good points
        good_new = p1[st==1]
        good_old = p0[st==1]

        good_new , good_old = filter_by_essential(good_new , good_old, 500/640*frame.shape[1], (frame.shape[1]/2, frame.shape[0]/2))

        tracked_region_mask = np.zeros(frame_gray.shape, dtype=np.uint8) + 255
        parallax = 0
        count = 0
        for i,(new,old) in enumerate(zip(good_new,good_old)):
            a,b = new.ravel()
            c,d = old.ravel()
            parallax += abs(a-c) + abs(b-d)
            count += 1
            #mask = cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
            tracked_region_mask = cv2.circle(tracked_region_mask,(a,b),mask_radius,0,-1)
            frame = cv2.circle(frame,(a,b),10,(255,0,0),2)

        parallax = int(parallax/count)
        font = cv2.FONT_HERSHEY_SIMPLEX
        imgzi = cv2.putText(frame, str(parallax), (50, 50), font, 1.2, (255, 255, 255), 2)

        if(parallax > para_interval):
            # detect new features to track
            p_new = cv2.goodFeaturesToTrack(frame_gray, mask = tracked_region_mask, **feature_params)
            for pt in p_new:
                frame = cv2.circle(frame,(pt[0][0],pt[0][1]),10,(0,0,255),2)
            # Now update the previous frame and previous points
            p_0 = np.array(good_new.reshape(-1,1,2))
            p_new = np.array(p_new).astype(np.float32)
            p0 = np.concatenate((p_0, p_new), axis=0)
            old_gray = frame_gray.copy()

            if(image_id%interval == 0):
                scale = 0.5
                frame_save = cv2.resize(frame_save, (int(frame_save.shape[1]*scale), int(frame_save.shape[0]*scale)));
                cv2.imwrite(save_folder+"/"+str(image_id)+'.jpg', frame_save)
            image_id += 1


        if(b_show):
            scale = 0.5
            img = cv2.resize(frame, (int(frame.shape[1]*scale), int(frame.shape[0]*scale)));
            cv2.imshow("Food table Video", img)
            tracked_region_mask = cv2.resize(tracked_region_mask, (int(frame.shape[1]*scale), int(frame.shape[0]*scale)));
            cv2.imshow("tracked_region_mask", tracked_region_mask)
            cv2.waitKey(int(1000/int(fps)))

    if(b_show):
        cv2.destroyAllWindows()
