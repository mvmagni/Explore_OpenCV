import cv2
 
# initialize the video stream
cap = cv2.VideoCapture('myvideo.mp4')
 
# start the server
rtspServer = cv2.VideoWriter_fourcc(*'MJPG')
outStream = cv2.VideoWriter('rtsp://localhost:8080/live.sdp', rtspServer, 25.0, (640,480))
 
# loop over frames from the video file
while(cap.isOpened()):
    # grab the frame from the video file
    ret, frame = cap.read()
 
    # check if the frame is None
    if frame is None:
        break
 
    # write the frame to the output stream
    outStream.write(frame)
 
    # show the frame
    cv2.imshow('frame',frame)
 
    # wait for 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
 
# release resources
cap.release()
outStream.release()
cv2.destroyAllWindows()