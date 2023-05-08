

import time
import pyrealsense2 as rs
import numpy as np
import cv2


class Camera(object):

    def __init__(self, width=1280, height=720, fps=30):
        self.width = width
        self.height = height
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, fps)
        self.config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, fps)
        # self.align = rs.align(rs.stream.color) # depth2rgb
        self.pipeline.start(self.config)

    def get_frame(self):
        frames = self.pipeline.wait_for_frames()

        align_to = rs.stream.color
        align = rs.align(align_to)
        aligned_frames = align.process(frames)

        aligned_depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        colorizer = rs.colorizer()
        depthx_image = np.asanyarray(aligned_depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        colorizer_depth = np.asanyarray(colorizer.colorize(aligned_depth_frame).get_data())
        return color_image, depthx_image, colorizer_depth

    def release(self):
        self.pipeline.stop()


if __name__ == '__main__':

    video_path = "C:\\Users\\akiso\\Desktop\\M\\rgb_data\\{int(time.time())}.mp4"
    video_depth_path = "C:\\Users\\akiso\\Desktop\\M\\depth_data\\{int(time.time())}_depth.mp4"
    video_depthcolor_path = "C:\\Users\\akiso\\Desktop\\M\\depthcolor_data\\{int(time.time())}_depthcolor.mp4"
    video_depthcolor_camera_path = "C:\\Users\\akiso\\Desktop\\M\\camera_colordepth\\{int(time.time())}_depthcolor.mp4"

    fps, w, h = 30, 1280, 720
    mp4 = cv2.VideoWriter_fourcc(*'mp4v')
    wr = cv2.VideoWriter(video_path, mp4, fps, (w, h), isColor=True)
    wr_depth = cv2.VideoWriter(video_depth_path, mp4, fps, (w, h), isColor=False)
    wr_depthcolor = cv2.VideoWriter(video_depthcolor_path, mp4, fps, (w, h), isColor=True)
    wr_camera_colordepth = cv2.VideoWriter(video_depthcolor_camera_path, mp4, fps, (w, h), isColor=True)

    cam = Camera(w, h, fps)
    print(' s, q')
    flag_V = 0
    while True:
        color_image, depth_image, colorizer_depth = cam.get_frame()
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        images = np.hstack((color_image, depth_colormap))
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense', colorizer_depth)

        # print('ll')
        key = cv2.waitKey(1)
        if key & 0xFF == ord('s'):
            flag_V = 1
        if flag_V == 1:
            wr.write(color_image)
            wr_depth.write(depth_image)
            wr_depthcolor.write(depth_colormap)
            wr_camera_colordepth.write(colorizer_depth)
            print('...Recording...')
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            print('...Recorded...')
            break
    wr_depthcolor.release()
    wr_depth.release()
    wr.release()
    wr_camera_colordepth.release()
    cam.release()
    print(f'，Video be saved at：{video_path}')
