function ex = wienerFilter(y,h,sigma,gamma,alpha);
%
% ex = wienerFilter(y,h,sigma,gamma,alpha);
%
% Generalized Wiener filter using parameter alpha. When
% alpha = 1, it is the Wiener filter. It is also called
% Regularized inverse filter.
%
% Reference: Richb's paper
% Created: Tue May 4 16:24:06 CDT 1999, Huipin Zhang

N = size(y,1);
Yf = fft(y);
Hf = fft(h,N);
Pyf = abs(Yf).^2/N^2;

% direct implementation of the regularized inverse filter, 
% when alpha = 1, it is the Wiener filter
% Gf = conj(Hf).*Pxf./(abs(Hf.^2).*Pxf+alpha*sigma^2);
%
% Since we don't know Pxf, the following 
% handle singular case (zero case)
sHf = Hf.*(abs(Hf)>0)+1/gamma*(abs(Hf)==0);
iHf = 1./sHf;
iHf = iHf.*(abs(Hf)*gamma>1)+gamma*abs(sHf).*iHf.*(abs(sHf)*gamma<=1);

Pyf = Pyf.*(Pyf>sigma^2)+sigma^2*(Pyf<=sigma^2);
Gf = iHf.*(Pyf-sigma^2)./(Pyf-(1-alpha)*sigma^2);

% max(max(abs(Gf).^2)) % should be equal to gamma^2
% Restorated image without denoising
eXf = Gf.*Yf;
ex = real(ifft(eXf));

return
