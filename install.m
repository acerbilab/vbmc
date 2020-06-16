% MATLAB installation script for VBMC
%
% Copyright (c) by Luigi Acerbi 2018-2020

fprintf('Installing VBMC...\n');

me = mfilename;                                 % what is my filename
pathstr = fileparts(which(me));                 % get my location
addpath(pathstr);                               % add to the path

try
    failed_install_flag = savepath;             % save path
catch
    failed_install_flag = true;
end

if failed_install_flag
    fprintf('Installation error: could not save path.\n\n');
    fprintf('You need to manually add VBMC''s installation folder to your MATLAB search path (and save it).\n'); 
    fprintf('See the <a href="https://www.mathworks.com/help/matlab/matlab_env/add-remove-or-reorder-folders-on-the-search-path.html">MATLAB documentation</a> for more information.\n'); 
    fprintf('Note that in Linux systems, e.g. Ubuntu, you need read/write permission to save the MATLAB path (see <a href="https://www.mathworks.com/matlabcentral/answers/95731-why-is-my-modified-path-not-saved-in-matlab">here</a>).\n'); 
else
    fprintf('Installation successful!\n');
    type([pathstr filesep 'docs' filesep 'README.txt']);
    fprintf('\n');
end

clear me pathstr