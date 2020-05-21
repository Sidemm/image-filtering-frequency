# image-filtering-frequency
Implementation of a low pass and high pass filter to use in hybrid image construction and practice of DFT Spectrum method.

Images are created with extra pixels around the corners. The new image is used to implement the image filtering equation;  
g[m, n] = ∑  h[i, j] f[m +i, n + j]  
The filter was multiplied with the image’s corresponding pixels to calculate the filtered value of every pixel of the image.

Filter function is used to find low frequent and high frequent images after applying low pass filter and 
high pass filter. The cat picture was chosen to apply high pass filter and sharp changes were more explicit than the dog’s image 
after low pass filtering. The result in dog’s image was more blurry, since only the small frequency changes were filtered. 
High pass filter is chosen for fish image because frequency change interval is more broad than the submarine and their hybrid image 
was more successful than doing it the opposite way. 

DFT_Spectrum method is implemented such that a new image is created with optimal dft rows and columns and filled the borders with 
zeros using BORDER_CONSTANT. Than this image is passed to dft method and the complex matrix variable is obtained. To be able to calculate 
the magnitude iit is splitted to its real and imaginary parts and its logarithm scale is calculated with the given formula; 
log(1 + sqrt(Re(DFT(I))^2 + Im(DFT(I))^2)) 
 
After that the parts of the image are relocated so that the origin of DC value would be in the center. Top left, top right, 
bottom left and bottom right parts are cropped and put in different matrix. Then top left is swapped with bottom right and top right is 
swapped with bottom left. Low pass filtered image’s accumulation is more centered in contrast to the high pass filtered 
image’s accumulation, since it is dispersed towards the other parts.
