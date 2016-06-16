from skimage import io
from skimage import segmentation
img = io.imread("1.jpg")
msk = segmentation.felzenszwalb(img,scale=100,sigma=1.5)
boun = segmentation.mark_boundaries(img,msk)
io.imsave("seg.jpg",boun)
