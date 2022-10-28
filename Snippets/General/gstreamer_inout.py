import cv2

#gst = 'rtspsrc location=rtsp://radar:radar@172.20.0.114:554/live latency=300 ! decodebin ! videoconvert ! video/x-raw,format=BGR ! appsink drop=1'
#gst = 'rtspsrc location=rtsp://172.20.0.30:8554/video_stream latency=10 ! decodebin ! videoconvert ! video/x-raw,format=BGR ! appsink drop=1'
gst = 'rtspsrc location=rtsp://172.20.0.30:8554/unicast latency=10 ! decodebin ! videoconvert ! video/x-raw,format=BGR ! appsink drop=1'

# Variant for NVIDIA decoder that may be selected by decodebin:
# gst = 'rtspsrc location=rtsp://username:pasword@10.2.9.164:554/h264Preview_01_main latency=300 ! decodebin ! nvvidconv ! video/x-raw,format=BGRx ! videoconvert ! video/x-raw,format=BGR ! appsink drop=1'

cap = cv2.VideoCapture(gst,cv2.CAP_GSTREAMER)

# Output
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
fps = cap.get(cv2.CAP_PROP_FPS)
print(f'Src opened: {w}x{h} @ {fps}')

#gst_out = 'appsrc ! queue ! videoconvert ! videoscale ! video/x-raw,width=640,height=480 ! x264enc speed-preset=veryfast tune=zerolatency bitrate=800 ! rtspclientsink location=rtsp://localhost:8554/live'
#gst_out = 'appsrc ! videoconvert ! video/x-raw ! video/x-raw ! x264enc ! rtspclientsink location=rtsp://localhost:8554/live'
#gst_out = 'appsrc ! queue ! videoconvert ! video/x-raw ! x264enc ! video/x-h264 ! h264parse ! rtph264pay ! udpsink host=127.0.0.1 port=5000 sync=false'
#gst_out = 'appsrc ! queue ! videoconvert ! video/x-raw ! x264enc ! video/x-h264 ! h264parse ! rtph264pay ! tcpserversink host=127.0.0.1 port=5000 sync=false'
#gst_out = 'fdsrc ! h264parse ! rtph264pay config-interval=1 pt=96! gdppay ! tcpserversink host=127.0.0.1 port=8554'
#gst_out = 'appsrc ! queue ! videoconvert ! video/x-raw ! x264enc ! video/x-h264 ! h264parse ! rtph264pay ! rtspclientsink location=rtsp://localhost:8554/live'
#gst_out = 'appsrc ! queue ! videoconvert ! video/x-raw ! x264enc ! h264parse ! rtph264pay ! udpsink host="172.20.0.51" port="2500" sync=false',0,25.0,(640,480)
#gst_out = 'appsrc ! video/x-raw, format=BGR ! queue ! videoconvert ! video/x-h264, stream-format=byte-stream ! h264parse ! rtph264pay pt=96 config-interval=1 ! udpsink host=172.20.0.1 port=8554'
#out = cv2.VideoWriter(gst_out, fourcc, 20.0, (640, 480))

#gst_out = 'rtspsrc location=rtsp://127.0.0.1:5000/live latency=100 ! queue ! rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! videoscale ! video/x-raw,width=640,height=480 ! autovideosink'
#out = cv2.VideoWriter(gst_out, fourcc, 15.0, (int(h), int(w)))

while(cap.isOpened()):
  ret, frame = cap.read()
  if not ret:
    break
  cv2.imshow('frame', frame)
  #out.write(frame)
  if cv2.waitKey(1) & 0xFF == ord('q'):
    break

cv2.destroyAllWindows()
cap.release()

