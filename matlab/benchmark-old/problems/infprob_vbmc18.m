function probstruct = infprob_vbmc18(prob,subprob,noise,id,options)

% Problem names (ordered)
problist = {'lumpy','cigar','studentt','goris2015'};

% Initialize problem structure
if ischar(subprob); D = extractnum(subprob); else; D = subprob; end

switch prob
    case 'goris2015'
        subprobString = ['S' num2str(D)];     
    otherwise
        subprobString = [num2str(D) 'D'];
end

probstruct = initprob(prob,problist,'vbmc18',subprobString,id,D);
% [~,probstruct.Title] = prob;
probstruct.Title = probstruct.Prob;
probstruct.func = ['@(x_,probstruct_) infbench_' probstruct.Prob '(x_(:)'',probstruct_.ProbInfo)'];

probstruct.Noise = noise;
probstruct.NoiseEstimate = 0;       % Function is intrinsically not-noisy

%--------------------------------------------------------------------------
function probstruct = initprob(prob,problist,ProbSet,SubProb,id,D)
%INITPROB Initialize problem structure.

if isnumeric(prob); prob = problist{prob}; end
fun = str2func(['infbench_' prob]);
probstruct.ProbSet = ProbSet;
probstruct.Number = find(strcmp(problist,prob),1);
probstruct.Prob = prob;
probstruct.SubProb = SubProb;
probstruct.Id = id;
probstruct.ProbInfo = fun([],D);


