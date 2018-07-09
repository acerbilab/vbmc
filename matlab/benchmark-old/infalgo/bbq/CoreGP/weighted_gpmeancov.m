function [wm,wC,covvy]=weighted_gpmeancov(rho,XStar,XData,covvy,flag,samples)

if isempty(XStar)
    wm=zeros(0,1);
    wC=zeros(0,1);
    return;
end

load_preds = nargin>4 && strcmp(flag,'load');
store_preds = nargin>4 && strcmp(flag,'store');

if nargin<6
    samples = 1:numel(covvy.hypersamples);
end

NStar=size(XStar,1);
wm=zeros(NStar,1);
wC=zeros(NStar,1);
num_samples=length(samples);
for sample_ind=1:num_samples
    sample = samples(sample_ind);
    if load_preds
        m=covvy.hypersamples(sample).mean_star;
        C=covvy.hypersamples(sample).C_star;    
    else
        [m,C] = gpmeancov(XStar,XData,covvy,sample);
        if store_preds
            covvy.hypersamples(sample).mean_star=m;
            covvy.hypersamples(sample).C_star=C;
            covvy.hypersamples(sample).SD_star=sqrt(diag(C));      
        end
    end
%     Cs(sample)=C(1);
%     ms(sample)=m(1);
    wm=wm+rho(sample)*m;
    wC=wC+rho(sample)*(diag(C)+m.^2);
end
wC=wC-wm.^2;
