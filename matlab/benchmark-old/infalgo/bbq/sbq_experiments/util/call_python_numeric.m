function result = call_python_numeric( script, args)
% Calls python, but converts arguments to strings,
% and converts the return value back to float.
%
% David Duvenauad
% March 2012
% ======================

cell_args = cell(length(args),1);
for i = 1:length(args)
    cell_args{i} = num2str(args(i));
end
result = str2double(python( script, cell_args{:} ));
