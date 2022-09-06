from datetime import datetime

# This is just a scratchpad to play with different functions outside of main files
frame_count=30
extension='jpg' 
date_format='%Y-%m-%d_%H-%M-%S'

print({frame_count})
print({extension})
print({date_format})

#fileName="test"
fileName = f"             {datetime.today().strftime(f'{date_format}_f{frame_count}.{extension}')}"
#fileName = f"{directory}/{datetime.today().strftime(f'{date_format}_f{frame_count}.{extension}')}"
print(fileName)








