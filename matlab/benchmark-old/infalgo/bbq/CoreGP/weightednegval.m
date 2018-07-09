function [f,g,H]=weightednegval(rho,XStar,covvy,varargin)

NStar=size(XStar,1);
f=zeros(NStar,1);
g=0;
H=0;
switch nargout 
    case {0,1}
        for sample=1:numel(covvy.hypersamples)   
                    fi=negval(XStar,covvy,sample,varargin{:});
                    f=f+rho(sample)*fi;
        end
    case 3
        for sample=1:numel(covvy.hypersamples)   
                    [fi,gi,Hi]=negval(XStar,covvy,sample,varargin{:});
                    f=f+rho(sample)*fi;
                    g=g+rho(sample)*gi;
                    H=H+rho(sample)*Hi;
        end
    case 2
        for sample=1:numel(covvy.hypersamples)   
                    [fi,gi]=negval(XStar,covvy,sample,varargin{:});
                    f=f+rho(sample)*fi;
                    g=g+rho(sample)*gi;
        end
end