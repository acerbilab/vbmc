function covvy = import_likelihood_vector(covvy, logL)
  
  if length(logL) ~= numel(covvy.hypersamples)
    disp('Vector is incorrect size')
    return;
  else
    for i = 1:numel(covvy.hypersamples)
      covvy.hypersamples(i).logL = logL(i);
    end
  end
  