% A silly function to make sure that hyperparameters don't go off to
% infinity.
%
% Should probably be handled somewhere else in GP inference.
%
%
% David Duvenaud
% May 2011
% ================================

function [nlz, dnlz] = penalized_gp(hyp, varargin)

[nlz, dnlz] = gp(hyp, varargin{:});

hyp = unwrap(hyp);
dnlz2 = unwrap(dnlz);

scale = 10;
polynomial = 10;
for i = 1:length(hyp)
    nlz = nlz + (hyp(i)/scale)^polynomial;
    dnlz2(i) = dnlz2(i) + polynomial*((hyp(i)/scale)^(polynomial-1))/scale;
end

dnlz = rewrap( dnlz, dnlz2);

end

function [s v] = rewrap(s, v)    % map elements of v (vector) onto s (any type)
if isnumeric(s)
  if numel(v) < numel(s)
    error('The vector for conversion contains too few elements')
  end
  s = reshape(v(1:numel(s)), size(s));            % numeric values are reshaped
  v = v(numel(s)+1:end);                        % remaining arguments passed on
elseif isstruct(s) 
  [s p] = orderfields(s); p(p) = 1:numel(p);      % alphabetize, store ordering
  [t v] = rewrap(struct2cell(s), v);                 % convert to cell, recurse
  s = orderfields(cell2struct(t,fieldnames(s),1),p);  % conv to struct, reorder
elseif iscell(s)
  for i = 1:numel(s)             % cell array elements are handled sequentially
    [s{i} v] = rewrap(s{i}, v);
  end
end
end

function v = unwrap(s)   % extract num elements of s (any type) into v (vector) 
v = [];   
if isnumeric(s)
  v = s(:);                        % numeric values are recast to column vector
elseif isstruct(s)
  v = unwrap(struct2cell(orderfields(s))); % alphabetize, conv to cell, recurse
elseif iscell(s)                                      % cell array elements are
  for i = 1:numel(s), v = [v; unwrap(s{i})]; end         % handled sequentially
end                                                   % other types are ignored
end