if ~isfield(covvy,'debug')
    covvy.debug = false;
end
debug = covvy.debug;

no_change = true;
if (debug); disp('no change'); end
covvy.lastHyperSampleMoved=[];
toMove = covvy2.lastHyper2SamplesMoved;
covvy.lastHyper2SamplesMoved = toMove; 

for i = toMove	
    covvy.hyper2samples(i).hyper2parameters = covvy2.hyper2samples(i).hyper2parameters;
end
