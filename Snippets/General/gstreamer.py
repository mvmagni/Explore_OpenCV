import cv2
gst = 'rtspsrc location=rtsp://radar:radar@172.20.0.114:554/live latency=300 ! decodebin ! videoconvert ! video/x-raw,format=BGR ! appsink drop=1'

# Variant for NVIDIA decoder that may be selected by decodebin:
# gst = 'rtspsrc location=rtsp://username:pasword@10.2.9.164:554/h264Preview_01_main latency=300 ! decodebin ! nvvidconv ! video/x-raw,format=BGRx ! videoconvert ! video/x-raw,format=BGR ! appsink drop=1'

cap = cv2.VideoCapture(gst,cv2.CAP_GSTREAMER)
while(cap.isOpened()):
  ret, frame = cap.read()
  if not ret:
    break
  cv2.imshow('frame', frame)
  if cv2.waitKey(1) & 0xFF == ord('q'):
    break

cv2.destroyAllWindows()
cap.release()