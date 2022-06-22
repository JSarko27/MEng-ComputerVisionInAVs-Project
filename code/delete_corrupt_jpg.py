#!/usr/bin/env python3

import glob
import os
from multiprocessing import Pool
from PIL import Image

def CheckOne(f):
    try:
        im = Image.open(f)
        im.verify()
        im.close()
        # DEBUG: print(f"OK: {f}")
        return
    except (IOError, OSError, Image.DecompressionBombError):
        # DEBUG: print(f"Fail: {f}")
        return f

if __name__ == '__main__':
    # Create a pool of processes to check files
    p = Pool()

    # Create a list of files to process
    files = [f for f in glob.glob(#"C:/Users/--- path to dataset----/*.jpg")]

    print(f"Files to be checked: {len(files)}")

    # Map the list of files to check onto the Pool
    result = p.map(CheckOne, files)

    # Filter out None values representing files that are ok, leaving just corrupt ones
    result = list(filter(None, result)) 
    print(f"Num corrupt files: {len(result)}")
   # corrupt_jpg = result
   #
    print(result)
    #i = 0
    #for file in gen_corrupt_txt:
     #   base = os.path.splitext(file)[0]
      #  gen_corrupt_txt[i] = base + '.txt'
        #print(gen_corrupt_txt[i])
       # i+=1
    #print(gen_corrupt_txt)
    #corrupt_txt = gen_corrupt_txt

    #corrupt_files = corrupt_jpg + corrupt_txt
    #print(corrupt_txt)
    

    
    #count = 0
    for y in result:
        files_loop = next(os.walk(#"C:/Users/---path to dataset"))
        #file_count = len(files_loop)
        if os.path.exists(y):
            os.remove(y)
       #     print("Removing " + y + "")
       #     print("File " + str(count) + " deleted")
        #    print(str(file_count) + "files left")
        #    count+=1
    print("Corrupt files have been deleted")
    print (len(result))

