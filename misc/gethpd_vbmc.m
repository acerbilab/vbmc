function [X_hpd,y_hpd,hpd_range] = gethpd_vbmc(optimState,options)
%GETHPD_VBMC Get high-posterior density dataset.

X = optimState.X(optimState.X_flag,:);
y = optimState.y(optimState.X_flag);
[N,D] = size(X);

% Subsample high posterior density dataset
[~,ord] = sort(y,'descend');
N_hpd = round(options.HPDFrac*N);
X_hpd = X(ord(1:N_hpd),:);
if nargout > 1
    y_hpd = y(ord(1:N_hpd));
end
if nargout > 2
    hpd_range = max(X_hpd)-min(X_hpd);
end

end