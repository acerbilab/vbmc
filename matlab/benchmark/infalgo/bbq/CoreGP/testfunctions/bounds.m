%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% bounds.m %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% function [u,v,fglob] = bounds(fcn)
% function [u,v,fglob] = bounds(fcn,def)
% defines box [u,v] (and dimension = size(u,1))
% for the functions fcn in {'bra','cam','gpr','hm3',
% 'hm6','ros','sh5','sh7','s10','sch','sgr','shu'}
% in addition, the global variables fglob, xglob and nglob are defined
%
% fglob        global minimum of fcn
% nglob        number of global minimizers in the box [u,v]
% xglob(:,i)   i-th global minimizers of test function fcn in [u,v]
%
% def          1 (default) to get the original bounds
%              0           to get randomly perturbed box bounds
%                          (each time different; for publishing tests,
%                          rather use the bounds from *.bnd)
%
function [u,v,fglob] = bounds(fcn,def)
global nglob xglob

if fcn == 'gpr'      % Goldstein-Price
  u = [-2; -2]; 
  v = [2; 2];
%  u = [-1.e6; -1.e6];
%  v = [2.e6; 2.e6];
  fglob = 3;
  xglob = [0.; -1.];
  nglob = 1;
elseif fcn == 'bra'  % Branin
%  u = [-1.e6; -1.e6];
%  v = [2.e6;  2.e6];
  u = [-5; 0];
  v = [10; 15];
  fglob = 0.397887357729739;
  xglob = [9.42477796   -3.14159265  3.14159265; 
           2.47499998   12.27500000  2.27500000];
  nglob = 3;  
elseif fcn == 'cam'  % Six-hump camel
  u = [-3; -2];
  v = [3; 2];
%  u = [-1.e6; -1.e6];
%  v = [ 2.e6;  2.e6];
  fglob = -1.0316284535;
  xglob = [ 0.08984201  -0.08984201;
           -0.71265640   0.71265640];
  nglob = 2;
elseif fcn == 'shu'  % Shubert
  n = 2;
%  u = [-1.e6; -1.e6];
%  v = [ 1.e6;  1.e6];
  u = [-10; -10];
  v = [10; 10];
  fglob = -186.730908831024;
  xglob = [
-7.08350658  5.48286415  4.85805691  4.85805691 -7.08350658 -7.70831382 -1.42512845 -0.80032121 -1.42512844 -7.08350639 -7.70831354  5.48286415  5.48286415  4.85805691 -7.70831354 -0.80032121 -1.42512845 -0.80032121; 
 4.85805691  4.85805681 -7.08350658  5.48286415 -7.70831382 -7.08350658 -0.80032121 -1.42512845 -7.08350639 -1.42512844  5.48286415 -7.70831354  4.85805691  5.48286415 -0.80032121 -7.70831354 -0.80032121 -1.42512845];
  nglob = 18;
elseif fcn == 'sh5'  % Shekel 5
  u = [0; 0; 0; 0];
  v = [10; 10; 10; 10];
  fglob = -10.1531996790582;
  xglob = [4; 4; 4; 4];
  nglob = 1;
elseif fcn == 'sh7'  % Shekel 7
  u = [0; 0; 0; 0];
  v = [10; 10; 10; 10];
  fglob = -10.4029405668187;
  xglob = [4; 4; 4; 4];
  nglob = 1;
elseif fcn == 's10'  % Shekel 10
  u = [0; 0; 0; 0];
  v = [10; 10; 10; 10];
  fglob = -10.5364098166920;
  xglob = [4; 4; 4; 4];
elseif fcn == 'hm3'  % Hartman 3
  u = [0; 0; 0];
  v = [1; 1; 1];
%  u = [-1.e6; -1.e6; -1.e6];
%  v = [1.e6; 1.e6; 1.e6];
  fglob = -3.86278214782076;
  xglob = [0.1; 0.55592003; 0.85218259];
  nglob = 1;
elseif fcn == 'hm6'  % Hartman 6
  n = 6;
  u = [0; 0; 0; 0; 0; 0];
  v = [1; 1; 1; 1; 1; 1];
  fglob = -3.32236801141551;
  xglob = [0.20168952;  0.15001069;  0.47687398;  0.27533243;  0.31165162;  0.65730054];
  nglob = 1;
else
  disp('fcn must be one of')
  disp('bra,cam,gpr,hm3,hm6,sh5,sh7,s10,sch,sgr,shu')
  error(['bounds for test function ',fcn,' not available'])
end

if nargin==1, return; end;
if def, return; end;

% perturbed box bounds
uold = u; vold = v;
u = u + 0.5*(rand(size(u)) - 0.5).*(v - u);
v = v + 0.5*(rand(size(u)) - 0.5).*(v - u);
ok = 0;
for i=1:nglob
  if u <= xglob(:,i) & xglob(:,i) <= v
    ok = 1; break
  end
end
if ~ok
  i = ceil(rand*nglob); 
  i1 = find(u>xglob(:,i));
  i2 = find(v<xglob(:,i));
  while i1 ~= [] | i2 ~= []
    u(i1) = uold(i1) + 0.5*(rand(size(u(i1))) - 0.5).*(vold(i1) - uold(i1));
    v(i2) = vold(i2) + 0.5*(rand(size(u(i2))) - 0.5).*(vold(i2) - uold(i2));   
    i1 = find(u>xglob(:,i));
    i2 = find(v<xglob(:,i)); 
  end 
end
u,v
