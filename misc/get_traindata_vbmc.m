function [X_train,y_train,s2_train,t_train] = get_traindata_vbmc(optimState,options)
%GETRAINDATA Get training data for building GP surrogate.

nvars = size(optimState.X,2);

X_train = optimState.X(optimState.X_flag,:);
y_train = optimState.y(optimState.X_flag);
if isfield(optimState,'S')
    s2_train = optimState.S(optimState.X_flag).^2;
else
    s2_train = [];
end

if options.NoiseShaping
    s2_train = noiseshaping_vbmc(s2_train,y_train,options);
end

if nargout > 3
    t_train = optimState.funevaltime(optimState.X_flag);
end



%         xxplot = (1:numel(y_train))';
%         [yyplot,ord] = sort(log(y_max - y_train + 1));
%         
%         X_train = X_train(ord,:);
%         y_train = y_train(ord);
%         
%         plot(xxplot,yyplot,'k-','LineWidth',1); hold on;
%         p = robustfit(xxplot,yyplot); p = fliplr(p');
%         pred = p(1).*xxplot + p(2);
%         plot(xxplot, pred,'b--','LineWidth',1);        
%         drawnow;

%         tail_idx = ceil(numel(y_train)*max(0.5,options.HPDFrac));
%         idx_start = find(yyplot(tail_idx:end) - pred(tail_idx:end) > 1,1);
%         if ~isempty(idx_start)
%             tail_idx = tail_idx + idx_start - 1;
%             [tail_idx,numel(y_train)]
%             yyplot(tail_idx:end) = min(pred(tail_idx:end),yyplot(tail_idx:end));            
%             y_train(tail_idx:end) = 1 + y_max - exp(yyplot(tail_idx:end));
%         end

end
