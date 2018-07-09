function y = combcell(varargin)
%COMBCELL Create all combinations of cell arrays.
%
%  COMBCELL(A1,A2,...) takes any number of inputs A, where each Ai is a 
%  cell array with Ni elements, and return a cell array Y of (N1*N2*...) 
%  elements. The elements of Y consist of cell arrays that contain all 
%  combinations found by combining one element from each Ai.
%
%  Inspired by COMBVEC by Mark Beale.
%
%  See also COMBVEC.

if isempty(varargin)
    y = [];
else    
    ncells = length(varargin);
    for i = 1:ncells
        if ~iscell(varargin{i}); varargin{i} = {varargin{i}}; end;
    end
    
    % Create first a matrix that indexes all elements
    for i = 1:ncells; m{i} = 1:length(varargin{i}); end    
    M = m{1};
    for i=2:ncells
        z = m{i};
        M = [copy_blocked(M,size(z,2)); copy_interleaved(z,size(M,2))];
    end 
    
    M = M';
        
    % Assign arrays
    for i = 1:size(M,1)
        for j = 1:ncells
            y{i}{j} = varargin{j}{M(i,j)};
        end        
    end
end

%--------------------------------------------------------------------------
function c = copy_blocked(a,b)

[mr,mc] = size(a);
c = zeros(mr,mc*b);
ind = 1:mc;
for i=[0:(b-1)]*mc; c(:,ind+i) = a; end

%--------------------------------------------------------------------------
function c = copy_interleaved(a,b)

[mr,mc] = size(a);
c = zeros(mr*b,mc);
ind = 1:mr;
for i=[0:(b-1)]*mr; c(ind+i,:) = a; end
c = reshape(c,mr,b*mc);