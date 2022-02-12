import cv2
import numpy as np
import os
import sys
import argparse
import logging
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans 
from tqdm import tqdm
import multiprocessing as mp
import collections
from scipy.spatial import distance
import time

logging.basicConfig(format='%(asctime)s - %(message)s')
#logging.basicConfig(level=logging.INFO)

def parseArgs():
    """parseArgs

    Returns: None
    """    
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', default='0', type=str, 
    help="Video input. Enter either a number for a webcam device or path to a video file. Example: python main.py --video test.mp4")
    parser.add_argument('--output', default='output.mp4', type=str,
    help="Video output path. Enter a string to the path where the video output will be saved. Default is a .mp4 file. Example: python main.py --output out.mp4")
    parser.add_argument('--depthai', action='store_true', default=False,
    help="Optional flag only used for DepthAI devices. Flag is false by default. After enabling program will import depthai module.")
    parser.add_argument('--noweight', action='store_true', default=False)
    return parser.parse_args()

class emojiCam:
    def __init__(self, emojiPicPath='./twitter', doWeight=True) -> None:
        self.emojiMap, self.emojis = self.__genEmojiMap__(emojiPicPath)
        self.emojis = self.__prepImages__(self.emojis)
        self.doWeight = doWeight

    def __call__(self, frame):
        """Takes in image frame and pixelizes it and maps it to emojis

        Args:
            frame (numpy ndarray): Image frame. Assuming to be (h x w x c) in shape

        Returns:
            numpy ndarray: The constructed emoji canvas
        """        
        canvas = np.zeros(frame.shape).astype(np.uint8)
        h,w,_ = frame.shape
        _frame = self.__reduceFilter__(frame, 6)
        for i in range(0,h,24):
            for j in range(0,w,24):
                segment = _frame[i:i+24,j:j+24,:][0,0]
                segment_block = _frame[i:i+24,j:j+24,:]
                #segment_mix = [segment for i in range(24*24)]
                emojiID = int(self.emojiMap[segment[0], segment[1], segment[2]])
                if emojiID == -1:
                    emo = self.emojis[-1]
                else:
                    emo = self.emojis[emojiID]

                if self.doWeight:
                    canvas[i:i+24, j:j+24, :] = cv2.addWeighted(emo, 0.7, segment_block, 0.3, 0.0)
                else:
                    canvas[i:i+24, j:j+24, :] = emo # no weight
        return canvas

    def __genEmojiMap__(self, emojiPicPath):
        """Function generates a large mapping matrix where each coordinate represents a RGB value. Each value of each coordinate is the id of a given emoji

        Args:
            emojiPicPath (str): 

        Returns:
            numpy ndarray: the mapping array
            list: a list of all emoji images with transparency channel removed and mapped to black
        """        
        
        for _,_,f in os.walk(emojiPicPath): pass
        imgList = [cv2.imread(os.path.join(emojiPicPath, _f), cv2.IMREAD_UNCHANGED) for _f in f]
        
        # create color grid:
        # subdivide into greater sections
        #rgbGrid = RGBTable(len(imgList))
        rgbGrid = np.ndarray((256,256,256))
        rgbGrid.fill(-1)
        rgbConf = np.ndarray((256,256,256))
        rgbConf.fill(0.0)

        # generate feature maps of each image:
        featureList = self.__cleanImages__(imgList)
        # conduct in kmeans on each image:
        featCenter = []
        featMeta = []
 
        #cache for development:
        if os.path.exists('rgbClusterCorrected.npy'):
            rgbGrid = np.load('rgbClusterCorrected.npy', allow_pickle=True)
        else:
            for feat in tqdm(featureList):
                feat = feat.astype(np.float32)
                num_px = feat.shape[0]
                clusters = KMeans(n_clusters=3, random_state=0).fit(feat)
                clusterCenter = clusters.cluster_centers_
                clusterLabels = clusters.labels_
                freq = collections.Counter(clusterLabels).items()
                freq = list(freq)
                freq = sorted(freq, key=lambda x: x[1], reverse=True)
                freq = [(f[0],f[1]/num_px) for f in freq]
                featCenter.append(clusterCenter)
                featMeta.append(freq)

            for i, (center, meta) in enumerate(tqdm(zip(featCenter, featMeta))):
                for j,(c, m) in enumerate(zip(center, meta)):
                    free_spaces = np.argwhere(rgbGrid == -1)
                    t = m[0]
                    rgb_idx = np.round(center[t]).astype(np.uint8)
                    current_val = rgbGrid[rgb_idx[0], rgb_idx[1], rgb_idx[2]]
                    
                    if current_val != -1:
                        # compare:
                        #print("compare!", current_val)
                        _dist = distance.cdist(np.expand_dims(rgb_idx,axis=0), free_spaces)
                        best_place = free_spaces[np.argmin(np.squeeze(_dist))] 

                        if rgbConf[best_place[0], best_place[1], best_place[2]] > m[1]:
                            rgbGrid[best_place[0], best_place[1], best_place[2]] = i
                            rgbConf[best_place[0], best_place[1], best_place[2]] = m[1]
                    else:
                        rgbGrid[rgb_idx[0], rgb_idx[1], rgb_idx[2]] = i
                        rgbConf[rgb_idx[0], rgb_idx[1], rgb_idx[2]] = m[1]
            
            free_spaces = np.argwhere(rgbGrid == -1)
            used_spaces = np.argwhere(rgbGrid != -1)
            print("Leftover spaces:", free_spaces.shape)
            for i in tqdm(range(0, free_spaces.shape[0])):
                fsp = free_spaces[i]
                ftu_dist = distance.cdist(np.expand_dims(fsp,axis=0), used_spaces)
                nearest_used = np.argmin(ftu_dist.squeeze())
                nu = used_spaces[nearest_used]
                rgbGrid[fsp[0], fsp[1], fsp[2]] = rgbGrid[nu[0], nu[1], nu[2]]
            rgbGrid = np.save('rgbClusterCorrected.npy', rgbGrid)

        return rgbGrid, imgList

    def __cleanImages__(self, imgList):
        """Function takes in image list. Removes all the transparent pixels and returns a flattened list of pixels for clustering.

        Args:
            imgList (list): A list of numpy images in shape of (h x w x c) where c=4

        Returns:
            list: a list of flattened pixels per image 
        """        
        cleanedImageList = []
        for i, img in enumerate(imgList):
            mask = img[:,:,-1] == 255
            img = img[:,:,:-1]
            img = img[mask,:]
            #img = np.expand_dims(img, axis=0)
            cleanedImageList.append(img)
        return cleanedImageList

    def __prepImages__(self, imgList, resize=24):
        """Function remaps green pixels associated with transparent pixels on the 3rd channel to black.

        Args:
            imgList (list): List of images
            resize (int, optional): . Defaults to 24.

        Returns:
            [type]: [description]
        """        
        
        _imgList = []
        for img in imgList:
            img = img.astype(np.float32)
            _multi = img[:,:,-1]/255.0
            _multi = np.array([_multi, _multi, _multi])
            _multi = np.transpose(_multi, (1,2,0))
            img[:,:,:3] *= _multi
            img = img[:,:,:3].astype(np.uint8)
            img = cv2.resize(img, (resize, resize))
            _imgList.append(img)
        return _imgList

    def __clusterImages__(self, featureMaps):
        return 0

    def __reduceFilter__(self, image, kernelSize):
        h,w,_ = image.shape
        cw,ch = (w//24,h//24)
        # Resize input to "pixelated" size
        temp = cv2.resize(image, (cw, ch), interpolation=cv2.INTER_LINEAR)
        # Initialize output image
        output = cv2.resize(temp, (w, h), interpolation=cv2.INTER_NEAREST)
        #image = cv2.filter2D(image, ddepth=-1, kernel=kernelSize)
        return output

if __name__ == '__main__':
    args = parseArgs()

    if args.depthai:
        import depthai as dai
        # Create pipeline
        pipeline = dai.Pipeline()
        
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        videoOut = cv2.VideoWriter(args.output, fourcc, 30, (1920,1080))

        # Define source and output
        camRgb = pipeline.create(dai.node.ColorCamera)
        xoutVideo = pipeline.create(dai.node.XLinkOut)

        xoutVideo.setStreamName("video")

        # Properties
        camRgb.setBoardSocket(dai.CameraBoardSocket.RGB)
        camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        videoSize = (1920,1080)
        camRgb.setVideoSize(videoSize[0], videoSize[1])

        xoutVideo.input.setBlocking(False)
        xoutVideo.input.setQueueSize(1)

        # Linking
        camRgb.video.link(xoutVideo.input)
        emoji = emojiCam(doWeight= not args.noweight)
        
        # Connect to device and start pipeline
        with dai.Device(pipeline) as device:

            video = device.getOutputQueue(name="video", maxSize=1, blocking=False)

            while True:
                try:
                    t1 = time.time()

                    videoIn = video.get()
                    rgbFrame = videoIn.getCvFrame()
                    out_frame = emoji(rgbFrame)
                    
                    t2 = time.time()    
                    logging.warn("Frame - FPS: %f"%(1/(t2-t1)))
                    cv2.imshow('Frame Viewer', out_frame)
                    cv2.waitKey(1)
                    videoOut.write(out_frame)
                except KeyboardInterrupt as e:
                    print("Finishing up recording...")
                    videoOut.release()
                    exit(1)
    else:
        try: 
            vid_in = int(args.video)
        except ValueError as e:
            vid_in = args.video
        videoIn = cv2.VideoCapture(vid_in)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        videoOut = cv2.VideoWriter(args.output, fourcc, 30, (1920,1080))

        emoji = emojiCam(doWeight= not args.noweight)
        
        while(True):
            ret, frame = videoIn.read()
            if not ret: break
            out_frame = emoji(frame)
            cv2.imshow('Frame View', out_frame)
            cv2.waitKey(1)
            videoOut.write(out_frame)

        videoIn.release()
        videoOut.release()