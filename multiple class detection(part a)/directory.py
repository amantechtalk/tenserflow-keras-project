import os
 
# Directory 
# Parent Directory path
parent_dir = r"C:\Users\amank\OneDrive\Desktop\intershila\vali"
 
# Path
for x in range ( 1,223):
   path = os.path.join(parent_dir, str(x))
 
   print(path)
   os.mkdir(path)
