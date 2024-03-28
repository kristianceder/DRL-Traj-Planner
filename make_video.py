import os
import glob
import pathlib
import time

import cv2 # type: ignore
import numpy as np

'''
This program reads images from the given directory and forms a video accordingly.
'''
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[0]

def reorder(img_list):
   """Reorder the image list to ensure the right order"""
   new_img_list = ['place_holder']*len(img_list)
   for img_name in img_list:
       idx = img_name.split('/')[-1]
       idx = idx.split('-')[1]
       idx = int(idx.split('.')[0])
       new_img_list[idx-1] = img_name
   return new_img_list

def imgs_to_video(img_list: list[str], video_path: str, keep_first_frame:int=1, keep_last_frame:int=1, fps:int=5):
   """Convert images to video"""
   frame_path_list = [img_list[0]]*keep_first_frame + img_list + [img_list[-1]]*keep_last_frame
   height, width, channel = cv2.imread(img_list[0]).shape
   out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc('M','J','P','G'), fps=fps, frameSize=(width, 505)) # type: ignore
   for i, fr in enumerate(frame_path_list):
       frame = cv2.imread(fr)
       out.write(frame[:505,:width])
       try:
           cv2.imshow('Preview', frame)
       except cv2.error:
           print(f'Video {video_path} is done!')
           break
       if cv2.waitKey(1) & 0xFF == ord('q'):
           break
   out.release()
   cv2.destroyAllWindows()

def main(img_dir, video_path, img_fmt:str='png'):
   img_list = glob.glob(img_dir+'/*.'+img_fmt)
   img_list = reorder(img_list) # Reorder to ensure right order
   imgs_to_video(img_list, video_path, keep_first_frame=5, keep_last_frame=5, fps=5)


if __name__ == '__main__':
   scene = '132'
#    img_relative_dir_list = [f'Demo/{scene}/mpc', f'Demo/{scene}/ddpg', f'Demo/{scene}/hyb']
   img_relative_dir_list = [f'Demo/{scene}/mpc']
   video_relative_path_list = [f'{img_dir}.avi' for img_dir in img_relative_dir_list]
   for img_r_dir, video_r_path in zip(img_relative_dir_list, video_relative_path_list):
       img_dir = os.path.join(PROJECT_ROOT, img_r_dir)
       video_path = os.path.join(PROJECT_ROOT, video_r_path)
       main(img_dir, video_path)
   print('Done!')