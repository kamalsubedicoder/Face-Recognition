import cv2
import os

def load_images_from_folder(folder):
    num=-2
    images = []
    for path,subdirnames,filenames in os.walk(folder):
        
        n=0
        num=num+1
        for filename in filenames:
            
            if filename.startswith("."):    #incase of error,skip that file
                print("skipping system file")
                continue
            img_path=os.path.join(path,filename)#create image path by joining filename and path
            print(img_path)
            img=cv2.imread(img_path)
            if img is None:
                print ("Not Loaded Properly")
                continue
            
            #print("Hello"+ str(num)+" "+str(n))
            blur = cv2.GaussianBlur(img,(5,5),0) # Add Gaussian filter to image    
            img_median = cv2.medianBlur(blur, 3) # Add median filter to image

            n+=1
    
            cv2.imwrite(r"C:\Users\Kieron subedi\Image\train-images\\"+str(num)+r"\result" + str(n) + ".jpg",img_median)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    return images

folder=r"C:\Users\Kieron subedi\Image\dataset"
load_images_from_folder(folder)
print("Pre-processing on dataset complete. ")
