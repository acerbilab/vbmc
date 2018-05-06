function [x,ok] = extractnum(s)
%EXTRACTNUM Convert any number within a string matrix to a numeric array.
%   X = EXTRACTNUM(S) converts a character array representation of a matrix 
%   of numbers to a numeric matrix, ignoring any non-numeric character 
%   except for '.','-', and '+'.
%
%   If the string S does not represent a valid number or matrix,
%   EXTRACTNUM(S) returns the empty matrix.  [X,OK]=EXTRACTNUM(S) will
%   return OK=0 if the conversion failed.
%
%   See also STR2NUM.

if ~ischar(s) || ~ismatrix(s)
   error('Requires string or character array input.');
end

if isempty(s)
    x = [];
    ok=false;
    return
end

s = s(:);
s(all(bsxfun(@ne, s, '0123456789-+.'),2)) = ' ';
[x,ok] = str2num(s(:)');