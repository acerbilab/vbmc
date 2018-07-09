function [result status] = python(varargin)
cmdString = '';
for i = 1:nargin
    thisArg = varargin{i};
    if isempty(thisArg) || ~ischar(thisArg)
        error('MATLAB:python:InputsMustBeStrings', 'All input arguments must be valid strings.');
    end
    if i==1
        if exist(thisArg, 'file')==2
            if isempty(dir(thisArg))
                thisArg = which(thisArg);
            end
        else
            error('MATLAB:python:FileNotFound', 'Unable to find Python file: %s', thisArg);
        end
    end
  if any(thisArg == ' ')
    thisArg = ['"', thisArg, '"'];
  end
  cmdString = [cmdString, ' ', thisArg];
end
errTxtNoPython = 'Unable to find Python executable.';
if isempty(cmdString)
  error('MATLAB:python:NoPythonCommand', 'No python command specified');
elseif ispc
  pythonCmd = 'C:\python27_64';
  cmdString = ['python' cmdString];  
  pythonCmd = ['set PATH=',pythonCmd, ';%PATH%&' cmdString];
  [status, result] = dos(pythonCmd)
else
  [status ignore] = unix('which python'); %#ok
  %if (status == 0)  % Why not try anyways?
    cmdString = ['python', cmdString];
    [status, result] = unix(cmdString);
  %else
  %   
  %  error('MATLAB:python:NoExecutable', errTxtNoPython);
  %end
end
if nargout < 2 && status~=0
  error('MATLAB:python:ExecutionError', ...
        'System error: %sCommand executed: %s', result, cmdString);
end
