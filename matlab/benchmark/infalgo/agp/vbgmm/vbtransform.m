function varargout = vbtransform(X,LB,UB,method)
%VBTRANSFORM Change of coordinates for constrained variables.

[d,n] = size(X);

if isempty(LB); LB = -Inf(d,1); end
if isempty(UB); UB = Inf(d,1); end

switch lower(method(1:3))
    
    case 'dir'  % Direct transform
        for i = 1:d
            % Bounded interval, use logit transform
            if isfinite(LB(i)) && isfinite(UB(i))
                X(i,:) = logiinv((X(i,:) - LB(i))./(UB(i)-LB(i)));

            % One-sided bounded intervals, use logarithmic transform
            elseif isfinite(LB(i))
                X(i,:) = log(X(i,:) - LB(i));

            elseif isfinite(UB(i))
                X(i,:) = log(UB(i,:) - X(i,:));                
            end            
        end
        varargout{1} = X;
        
    case 'inv'  % Inverse transform
        for i = 1:d
            % Bounded interval, use inverse logit transform
            if isfinite(LB(i)) && isfinite(UB(i))
                X(i,:) = LB(i) + (UB(i) - LB(i))*logicdf(X(i,:));
                
            % One-sided bounded intervals, use exponential transform
            elseif isfinite(LB(i))
                X(i,:) = exp(X(i,:)) + LB(i);
                
            elseif isfinite(UB(i))
                X(i,:) = UB(i,:) - exp(X(i,:));                
            end            
        end
        varargout{1} = X;
        
    case 'lgr'  % Log gradient
        lg = zeros(d,n);
        for i = 1:d
            % Bounded interval, use inverse logit transform
            if isfinite(LB(i)) && isfinite(UB(i))
                temp = exp(-X(i,:));
                lg(i,:) = (log(UB(i)-LB(i))-log1p(temp) + log(1-(1./(1+temp))));

            % One-sided bounded intervals, use exponential transform
            elseif isfinite(LB(i)) || isfinite(UB(i))
                lg(i,:) = X(i,:);
            end
        end
        varargout{1} = lg;
        
end