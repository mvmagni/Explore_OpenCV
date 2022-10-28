# import cv2 as cv
# import os

# print(f'{os.getcwd()}')
# alpha = 0.5
# # [load]
# src1 = cv.imread('Explore_OpenCV/resources/background.jpg')
# src2 = cv.imread('Explore_OpenCV/resources/HUD.jpg')
# # [load]
# if src1 is None:
#     print("Error loading src1")
#     exit(-1)
# elif src2 is None:
#     print("Error loading src2")
#     exit(-1)
# # [blend_images]
# beta = (1.0 - alpha)
# dst = cv.addWeighted(src1, alpha, src2, beta, 0.0)
# # [blend_images]
# # [display]
# cv.imshow('dst', dst)
# cv.waitKey(0)
# # [display]
# cv.destroyAllWindows()



# importing the module 
from pytube import YouTube 
  
# where to save 
SAVE_PATH = "D:/" #to_do 
  
# link of the video to be downloaded 
link="https://www.youtube.com/watch?v=yNjPL_LJrWs"
  
try: 
    # object creation using YouTube
    # which was imported in the beginning 
    yt = YouTube(link) 
except: 
    print("Connection Error") #to handle exception 
  
# filters out all the files with "mp4" extension 
#mp4files = yt.streams.filter(file_extension='mp4', resolution='720p') 
#mp4files = yt.streams.filter()
#mp4files = yt.streams.all()
#for res in mp4files:
#    print(res)
#print(type(mp4files))
#print(mp4files)

# #to set the name of the file
#yt.set_filename('IronMan_HallOfFame.mp4')  
yt.streams.get_highest_resolution().download(SAVE_PATH)
# # get the video with the extension and
# # resolution passed in the get() function 
# d_video = yt.get(mp4files[-1].extension,mp4files[-1].resolution) 
# try: 
#     # downloading the video 
#     d_video.download(SAVE_PATH) 
# except: 
#     print("Some Error!") 
# print('Task Completed!') 