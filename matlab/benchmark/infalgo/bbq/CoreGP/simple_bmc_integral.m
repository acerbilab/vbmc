function result = simple_bmc_integral(samples, values, prior)
% returns bmc estimate for integral int y(x) p(x) dx
% where samples = N-long vector is the sample x values
% values = N-long vector is the y values at those samples
% prior.mean, prior.SD and prior.type specify p(x).


inputs = 1:numel(prior);
widthfrac = 0.20;

for input=inputs
    type=prior(input).type;

    if strcmp(type,'discrete')


    else

        samples_input=samples(:,input);

        priorMean=prior(input).mean;
        priorSD=prior(input).SD;

        if ~all(~isnan([priorMean;priorSD]));
            % This inputeter is a dummy - ignore it
            continue
        end

        width=widthfrac*separation(samples_input);
        Ks=matrify(@(x,y) normpdf(x,y,width),samples_input,samples_input);
        cholKs=chol(Ks);


        switch lower(type)
            case 'real'
                ns = normpdf(samples_input,priorMean,sqrt(priorSD^2+width^2));
            case 'bounded'

            case 'mixture'
                mixtureWeights = prior(input).mixtureWeights;

                ns = 0;
                for i = 1:length(mixtureWeights)
                    ns = ns + mixtureWeights(i)*normpdf(samples_input,priorMean(i),sqrt(priorSD(i)^2+width^2));
                end
        end
        
        

    end
end

result = ns'*solve_chol(cholKs,values);


function s = separation(ls) 
if length(ls)<=1
    s=1;
else
    s=min(diff(sort(ls)));
end
