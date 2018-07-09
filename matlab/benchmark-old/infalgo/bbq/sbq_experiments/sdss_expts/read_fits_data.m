function [wavelengths, flux, noise_variance, redshift, is_quasar] = ...
      read_fits_data(filename, debug)

  measurements = fitsread(filename, 'BinTable', 1);

  flux                   = measurements{1};
  log_wavelengths        = measurements{2};
  inverse_noise_variance = measurements{3};
  
  to_keep_ind = (inverse_noise_variance ~= 0);

  wavelengths    = 10.^log_wavelengths(to_keep_ind);
  flux           = flux(to_keep_ind);
  noise_variance = 1 ./ inverse_noise_variance(to_keep_ind);

  measurements = fitsread(filename, 'BinTable', 2);

  ancillary_target_1 = measurements{15};
  ancillary_target_2 = measurements{16};
  redshift           = measurements{20};
  class_person       = measurements{21};
  object_type        = measurements{32};
  class              = measurements{35};

  occurs = @(x, pattern) (numel(strfind(x, pattern)) > 0);
  
  % the criteria used by sdss to determine quasarhood
  is_quasar = ((occurs(object_type, 'QSO')       || ...
               (occurs(object_type, 'GALAXY') &&    ...
                occurs(class      , 'QSO'))      || ...
               (class_person == 3)               || ...
               (bitand(ancillary_target_1, ...
                       hex2dec('ffc00000')) > 0) || ...
               (bitand(ancillary_target_2, ...
                       hex2dec('3b8')) > 0)));
  
  if ((nargin > 1) && debug)
    fprintf('loaded %s.\nredshift: %f\n', filename, redshift);
    if (is_quasar)
      fprintf('this object is a quasar.\n');
    else
      fprintf('this object is not a quasar.\n');
    end
  end
  
end