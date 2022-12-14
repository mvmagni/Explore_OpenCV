<!-- PROJECT LOGO -->
<div align="center">
  <a href="https://github.com/mvmagni/magritte">
    <img src="../resources/HUD.jpg" alt="Logo" width="500" height="300"/>
  </a>
    <p>
    <h2 align="center">OpenCV Exploration</h2>
    </p>

  
</div>


<!-- ABOUT THE PROJECT -->
## Raspberry Pi Setup

  - Raspberry Pi 4/4GB<br />
  - OpenCV 4.5.5
  - Webcam (not a pi camera)    
 
## Resources
<a href="https://www.raspberrypi.com/software/"> RaspberryPi Imager</a><br />
<a href="https://qengineering.eu/install-opencv-4.5-on-raspberry-pi-4.html">OpenCV on RaspberryPi</a><br />
<a href="https://raspberrypi-guide.github.io/electronics/using-usb-webcams">Webcam on pi (not pi camera)</a><br />
<a href="https://code.visualstudio.com/docs/setup/raspberry-pi"> VSCode on RaspberryPi </a><br />
<a href="https://fedoramagazine.org/use-opencv-on-fedora-linux-part-1/">OpenCV on Fedora</a><br />


## Notes
  
  - Using advanced options in pi imager (preset host, ssh, wifi) caused a boot loop
  - Reset image to base and configured afterwards
  - OpenCV script from [article](("https://qengineering.eu/install-opencv-4.5-on-raspberry-pi-4.html") works well. Includes qt on install.
  - OpenCV install takes about 1.5 hrs on a pi4
  - Legacy camera mode is being deprecated in raspbian. 
  - Webcam does not work with `raspistill` image capture
    - Webcam test `fswebcam -r 1920x1080 --no-banner /path/to/output/file.jpg`
  - Kill background on Pi for better remote, especially across multiple monitors



