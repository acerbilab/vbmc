required_options = {'plate', 'mjd', 'fiber'};
check_required_options;

if (options_defined)

  % below we subsample the data to every skip'th point
  skip = 1;

  % base directory where spectra are stored
  data_directory = '~/Code/gp-code-osborne/sbq_experiments/sdss_expts/';

  filename = @(plate, mjd, fiber) ...
             ([data_directory ...
               num2str(plate) '/spec-' ...
               num2str(plate) '-' ...
               num2str(mjd)   '-' ...
               num2str(fiber, '%04i') '.fits']);

  % this tends to work better but is faster and lamer, try if you'd like

  inference_method              = @infLaplace;
  likelihood                    = @likT;
  continuum_hyperparameters.lik = nan(2, 1);

%   likelihood                     = @likLaplace;
%   inference_method               = @infEP;
%   continuum_hyperparameters.lik  = nan;

  continuum_mean_function        = {@meanConst};
  continuum_hyperparameters.mean = nan;

  continuum_covariance_function  = {@covMaterniso, 3};
  continuum_hyperparameters.cov  = nan(2, 1);

  [wavelengths, flux, noise_variance, redshift, is_quasar] = ...
      read_fits_data(filename(plate, mjd, fiber), false);

  if (is_quasar)

    test_x  = wavelengths;
    train_x = wavelengths(1:skip:end);
    train_y = flux(1:skip:end);

    continuum_likelihood = @(sample) ...
        gp(rewrap(continuum_hyperparameters, sample), ...
           inference_method, continuum_mean_function, ...
           continuum_covariance_function, likelihood, train_x, train_y);
  else
    fprintf('the selected data are not observations of a quasar!\n');
  end
end