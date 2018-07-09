function [interp_cmap]=interpolate_cbrewer(cbrew_init, interp_method, ncolors)
% 
% INTERPOLATE_CBREWER - interpolate a colorbrewer map to ncolors levels
%
% INPUT:
%   - cbrew_init: the initial colormap with format N*3
%   - interp_method: interpolation method, which can be the following:
%               'nearest' - nearest neighbor interpolation
%               'linear'  - bilinear interpolation
%               'spline'  - spline interpolation
%               'cubic'   - bicubic interpolation as long as the data is
%                           uniformly spaced, otherwise the same as 'spline'
%   - ncolors=desired number of colors 
%
% Author: Charles Robert
% email: tannoudji@hotmail.com
% Date: 14.10.2011


% just to make sure, in case someone puts in a decimal
ncolors=round(ncolors);

% How many data points of the colormap available
nmax=size(cbrew_init,1);

% create the associated X axis (using round to get rid of decimals)
a=(ncolors-1)./(nmax-1);
X=round([0 a:a:(ncolors-1)]);
X2=0:ncolors-1;

z=interp1(X,cbrew_init(:,1),X2,interp_method);
z2=interp1(X,cbrew_init(:,2),X2,interp_method);
z3=interp1(X,cbrew_init(:,3),X2, interp_method);
interp_cmap=round([z' z2' z3']);

end