addpath(genpath('~/Code/gpml-matlab/'))

required_options = {'plate', 'mjd', 'fiber'};
check_required_options;

if (options_defined)

  % below we subsample the data to every skip'th point
  skip = 20;

  % base directory where spectra are stored
  data_directory = '~/Code/gp-code-osborne/sbq_experiments/sdss_expts/';

  filename = @(plate, mjd, fiber) ...
             ([data_directory ...
               num2str(plate) '/spec-' ...
               num2str(plate) '-' ...
               num2str(mjd)   '-' ...
               num2str(fiber, '%04i') '.fits']);
          
    cd(data_directory)
           
    load bests
    load priors

  % this tends to work better but is faster and lamer, try if you'd like

  inference_method              = @infLaplace;
  likelihood                    = @likT;
  continuum_hyperparameters.lik = nan(2, 1);
  hyperparameter_inds           = 1:5;
  
  total_hyperparameters.lik      = continuum_hyperparameters.lik;
  
%   inference_method               = @infEP;
%   likelihood                     = @likLaplace;
%   continuum_hyperparameters.lik  = nan;

%   hyperparameter_inds            = [1:2 4:5];

  continuum_mean_function        = {@meanConst};
  continuum_hyperparameters.mean = nan;
  dla_mean_function              = {@meanScale, {@meanDrift, {@meanVoightProfile}}};
  total_mean_function            = {@meanSum, {continuum_mean_function, dla_mean_function}};
  total_hyperparameters.mean     = nan(4, 1);
  
  continuum_covariance_function  = {@covMaterniso, 3};
  continuum_hyperparameters.cov  = nan(2, 1);
  dla_covariance_function        = {@covDrift, {@covMaterniso, 3}};
  total_covariance_function      = {@covSum, {continuum_covariance_function, dla_covariance_function}};
  total_hyperparameters.cov      = nan(6, 1);  
  
  [wavelengths, flux, noise_variance, redshift, is_quasar] = ...
      read_fits_data(filename(plate, mjd, fiber), false);

  if (is_quasar)

    test_x  = wavelengths;
    train_x = wavelengths(1:skip:end);
    train_y = flux(1:skip:end);

    continuum_likelihood = @(sample) ...
        -gp_likelihood(rewrap(continuum_hyperparameters, sample), ...
                       inference_method, continuum_mean_function, ...
                       continuum_covariance_function, likelihood, ...
                       train_x, train_y);

    continuum_prior = @(sample) ...
        -sum(arrayfun(@(i) normlike([mle_means(i), mle_stds(i)], ...
                                    sample(i)), hyperparameter_inds));
    
    continuum_hyperparameters = rewrap(continuum_hyperparameters, ...
            mle_means(hyperparameter_inds));
    [continuum_mle_hyperparameters, continuum_mle_probability] = ...
        minimize(continuum_hyperparameters, @gp_likelihood, 20, ...
                 inference_method, continuum_mean_function, ...
                 continuum_covariance_function, likelihood, train_x, train_y);
        
    fprintf('mle log likelihood (no dla): %f\n', -continuum_mle_probability(end));
  
    continuum_mle_point = unwrap(continuum_mle_hyperparameters);

    [~, ~, continuum_mle_mean, continuum_mle_variance] = ...
        gp_test(continuum_mle_hyperparameters, inference_method, ...
                continuum_mean_function, continuum_covariance_function, ...
                likelihood, train_x, train_y, test_x);
    
    continuum_mle_mean_value = @(x) interp1(test_x, continuum_mle_mean, x);
    
    transition_wavelength = 1215.688;
    convert_z = @(z) ((z + 1) * transition_wavelength);
    
    maximum_dla_location = convert_z(redshift);

% dla hyperparameters:
   %  1-6: covariance
   %      1: log input scale    (continuum)
   %      2: log output scale   (continuum)
   %      3: dla central lambda (dla)
   %      4: log dla width      (dla)
   %      5: log input scale    (dla)
   %      6: log ouput scale    (dla)
   %
   %  7-8: likelihood
   %      7: log nu - 1
   %      8: log noise
   %
   % 9-12: mean
   %      9: constant mean      (continuum)
   %     10: scaling for dla    (dla, determined)
   %     11: dla central lambda (dla, same as above)
   %     12: log dla width      (dla, same as above)

   
   load concordance

    dla_offset_prior_mean = mean(convert_z(concordance(:, 4)) - ...
        convert_z(concordance(:, 7)));
    dla_offset_prior_std = std(convert_z(concordance(:, 4)) - ...
        convert_z(concordance(:, 7)));
   

    dla_width_prior_mean = log(35);
    dla_width_prior_std = 0.4;
    
    sample_3 = bound(maximum_dla_location - dla_offset_prior_mean, min(train_x), maximum_dla_location);
    sample_4 = dla_width_prior_mean;

construct_dla_sample = @(sample) ...
      [mle_means(1:2),...
      sample_3, ...
      sample_4, ...
      sample,...
       continuum_mle_mean_value(sample_3),...
       sample_3,sample_4];
   
%    construct_dla_sample = @(sample) ...
%       [sample(1:2),bound(sample(3), min(train_x),
% maximum_dla_location), sample(4:9),...
%        continuum_mle_mean_value(bound(sample(3), min(train_x),
% maximum_dla_location)),...
%        bound(sample(3), min(train_x), maximum_dla_location),sample(4)];

%    dla_log_prior = ...
%        @(sample) ...
%        continuum_prior([sample(1:2); sample(7:9)]) + ...
%        -normlike([dla_offset_prior_mean, dla_offset_prior_std], ...
%             maximum_dla_location - bound(sample(3), min(train_x), maximum_dla_location)) + ...
%        -normlike([dla_width_prior_mean, dla_width_prior_std], ...
%                  sample(4));

   dla_log_likelihood = @(sample) -gp_likelihood( ...
       rewrap(total_hyperparameters, construct_dla_sample(sample)), ...
       inference_method, total_mean_function, total_covariance_function, ...
       likelihood, train_x, train_y);
  else
    fprintf('the selected data are not observations of a quasar!\n');
  end
end