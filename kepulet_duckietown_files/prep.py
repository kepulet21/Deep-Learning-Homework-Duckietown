# coding=utf-8
import cv2 as cv2
import numpy as np
import PIL as Image





def prep_image(image_):#preprocess one image
        
        test = image_
        
        img_height=60#target height
        img_width=80#target width
        #cv2.imwrite('raw.png',image_)
        image_ = cv2.resize(image_, None,fx=(img_width/image_.shape[1]), fy=(img_height/image_.shape[0]), interpolation = cv2.INTER_CUBIC)#resize to target shape 
        image_ = image_[ 20:60,0:80]#crop image (get processable data)
        image_ = cv2.resize(image_, None,fx=(1), fy=(30/20), interpolation = cv2.INTER_CUBIC)
        #image_ = cv2.cvtColor(image_, cv2.COLOR_BGR2RGB)#toRGB
        #cv2.imwrite('resized_raw.png',image_)
        img_red = image_[:,:,0]
        img_green = image_[:,:,1]
        img_blue = image_[:,:,2]
        
        lower_white = (0,0,70)#lower filter
        upper_white = (360,20 ,300 )#upper filter
        
        #lower_white = (0,0,150)#lower filter
        #upper_white = (170,20 ,300 )#upper filter
        hsv_ = cv2.cvtColor(image_, cv2.COLOR_RGB2HSV)#hsv
        mask_ = cv2.inRange(hsv_, lower_white, upper_white)#mask (one channel)
        #plt.imshow(mask_)
        #plt.show()
        red = cv2.bitwise_and(img_red, img_red, mask = mask_)
        #cv2.imwrite('red.png',red[:,:])
        #print("Red channel:")
        #plt.imshow(red)
        #plt.show()



        lower_orange = (10,30,170)#lower filter
        upper_orange = (30,200 ,300 )#upper filter
        hsv = cv2.cvtColor(image_, cv2.COLOR_RGB2HSV)#hsv
        mask = cv2.inRange(hsv, lower_orange, upper_orange)#mask (one channel)
        #plt.imshow(mask)
        #plt.show()
        blue = cv2.bitwise_and(img_blue, img_blue, mask = mask)
        #cv2.imwrite('blue.png',blue[:,:])
        #print("Blue cahnnel:")
   
        #plt.imshow(blue)
        #plt.show()


        lower_def = (0,0,0)#lower filter
        upper_def = (0,0 ,0 )#upper filter
        hsv__ = cv2.cvtColor(image_, cv2.COLOR_RGB2HSV)#hsv
        mask__ = cv2.inRange(hsv__, lower_def, upper_def)#mask (one channel)
        #plt.imshow(mask__)
        plt.show()
        green = cv2.bitwise_and(img_green, img_green, mask = mask__)
        #cv2.imwrite('green.png',green[:,:])
        #print("Green channel:")
        #plt.show()
        #plt.imshow(green)

        final = cv2.merge((red, green, blue))
        #for i in range(3):
        #    final[:,:,i] = final[:,:,i]/255.0 #normalize image pixels

        #v2.imwrite('Screenshot3.png',final)
        #plt.imshow(final)
        return final#return preprocessed image
        

        
        
        
def prep_images(self, images_, image_):
        if np.shape(images_)[2] < 15:
            x=np.empty((60,80,15))
            for i in range(5):
                for j in range(3):
                    x[:,:,i+j] = self.prep_image(image_)[:,:,j]
            return x
        else:
            #print(np.shape(images_))
            tmp = images_
            for i in range(15):
                images_[:,:,i] = tmp[:,:,i]
            for j in range(3):
                images_[:,:,-3+j] = self.prep_image(image_)[:,:,j]
            return images_
