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
