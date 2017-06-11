#!/home/jcenteno/anaconda3/envs/carnd-term1/bin/python
from moviepy.editor import VideoFileClip
from IPython.display import HTML
import numpy as np
import cv2
from moviepy.editor import VideoFileClip

from pipeline import *



def process_image(image):
    result = pipeline(image)
    return result


#clip1 = VideoFileClip("project_video.mp4")
clip1 = VideoFileClip("test_video.mp4")
#clip1 = VideoFileClip("project_video.mp4").subclip(4,24)
processed_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!

#output_name = 'video_output.mp4'
output_name = 'test_video_output.mp4'
#output_name = 'clip_video_output.mp4'

processed_clip.write_videofile(output_name, audio=False)