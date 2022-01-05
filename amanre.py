import re
import csv
import time
import shutil
import os


import csv
k=[]
# opening the CSV file
with open('train_labels.csv', mode ='r')as file:

#
 csvFile = csv.reader(file)

# displaying the contents of the CSV file
 for lines in csvFile:
    
    
    aman=' '.join(lines)
    str(aman)

    x = re.findall('[0-9]+', aman)
    l=len(x)
    try:
   
     for t in range(len(x)):
   
     
       path =r'C:\Users\amank\OneDrive\Desktop\intershila\Data\Data\validation'
       path1 =r'C:\Users\amank\OneDrive\Desktop\intershila\vali'
       src = os.path.join(path ,x[l-1]+str('.jpg'))
       dst = os.path.join(path1 ,x[t]+  str('/') +str(x[l-1]) + str('.jpg'))
       
      

       shutil.copyfile(src, dst)

     
    except:
      path="ll"
    
        
    

