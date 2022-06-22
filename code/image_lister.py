import glob

imglist = glob.glob("data/obj/*.jpg", recursive=False) # use for training set
#imglist = glob.glob("data/test/*.jpg", recursive=False) #use for validation set
#with open("data/test.txt", 'w', encoding = 'utf-8') as f: # use for validation set
with open("data/train.txt", 'w', encoding = 'utf-8') as f: # use for training set
    for img in imglist:
        img = img.replace("\\", "/")

        f.write(img + '\n')