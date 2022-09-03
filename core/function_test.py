# This is just a scratchpad to play with different functions outside of main files

import utils
from datetime import datetime

frame_count = 456
directory = f'd:/OBS_Recordings'

fileName = f"{directory}/{datetime.today().strftime(f'%Y-%m-%d_%H-%M-%S_f{frame_count}.png')}"
print (fileName)