clear;clf;

% Input and display the binary image
N = 256;
x = readbin('lenna.256',N,N);
figure(1)
imshow(x,gray(256))

% Blur the image, corrupt the image using WGN and display it
% h is the blurring filter, and sigma is the noise std
h = ones(4,4)/16;
sigma = 10;

Xf = fft2(x);
Hf = fft2(h,N,N);
y = real(ifft2(Hf.*Xf))+sigma*randn(N,N); % circular convolution
%y = filter2(h,x)+sigma*randn(N,N);	  % linear convolution

figure(2)
imshow(y,gray(256))

% restoration using generalized inverse filtering
gamma = 2;
eix = inverseFilter(y,h,gamma);
figure(3)
imshow(eix,gray(256))

% restoration using generalized Wiener filtering
gamma = 1;
alpha = 1;
ewx = wienerFilter(y,h,sigma,gamma,alpha);
figure(4)
imshow(ewx,gray(256))

PSNR = [psnr(y,x) psnr(eix,x) psnr(ewx,x)]
return
