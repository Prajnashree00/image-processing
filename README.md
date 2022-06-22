# image-processing
import cv2
img=cv2.imread('leaf1.jpg',0)
cv2.imshow('leaf1',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
![image](https://user-images.githubusercontent.com/97970956/174995803-db53dd6a-14fe-426c-9dcd-6cf0466c102e.png)









1.cover the image to URL code
from skimage import io
import matplotlib.pyplot as plt
url='https://i.natgeofe.com/n/3861de2a-04e6-45fd-aec8-02e7809f9d4e/02-cat-training-NationalGeographic_1484324_3x2.jpg'
image=io.imread(url)
plt.imshow(image)
plt.show()

output:
![image](https://user-images.githubusercontent.com/97970956/175013222-2b97c85a-cc09-4285-9763-08249fb4c856.png)



import cv2
import matplotlib.image as mping
import matplotlib.pyplot as plt
img=cv2.imread('ow.jpg')
plt.imshow(img)
plt.show()
![image](https://user-images.githubusercontent.com/97970956/175023065-853033e8-ec58-46ad-b5c6-2ba27e546234.png)


hsv_img=cv2.cvtColor(img,cv2.COLOR_RGB2HSV)
light_orange=(1,190,200)
dark_orange=(18,255,255)
mask=cv2.inRange(img,light_orange,dark_orange)
result=cv2.bitwise_and(img,img,mask=mask)
plt.subplot(1,2,1)
plt.imshow(mask,cmap="gray")
plt.subplot(1,2,2)
plt.imshow(result)
plt.show()
![image](https://user-images.githubusercontent.com/97970956/175023157-0320f0f4-eee6-4aa1-bd66-4c1aba56febb.png)



light_white=(0,0,200)
dark_white=(145,60,255)
mask_white=cv2.inRange(hsv_img,light_white,dark_white)
result_white=cv2.bitwise_and(img,img,mask=mask_white)
plt.subplot(1,2,1)
plt.imshow(mask,cmap="gray")
plt.subplot(1,2,2)
plt.imshow(result_white)
plt.show()

![image](https://user-images.githubusercontent.com/97970956/175023247-8a5cb492-e995-4afa-9c40-2d757a151820.png)


final_mask=mask+mask_white
final_result=cv2.bitwise_and(img,img,mask=final_mask)
plt.subplot(1,2,1)
plt.imshow(final_mask,cmap="gray")
plt.subplot(1,2,2)
plt.imshow(final_result)
plt.show()
![image](https://user-images.githubusercontent.com/97970956/175023337-725d8500-558c-4dee-b78a-28862519f347.png)

blur=cv2.GaussianBlur(final_result,(7,7),0)
plt.imshow(blur)
plt.show()
![image](https://user-images.githubusercontent.com/97970956/175023457-d139c3cf-e08e-428f-9951-74a154921e79.png)




