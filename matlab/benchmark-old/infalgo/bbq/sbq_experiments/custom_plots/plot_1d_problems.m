function plot_1d_problems( plotdir )

if nargin < 1
    plotdir = '~/Dropbox/papers/sbq-paper/figures/integrands/';
end
mkdir( plotdir );

%addpath(genpath(pwd));
problems = define_integration_problems();
num_problems = length(problems);

n = 1000;
for p_ix = 1:num_problems
    problem = problems{p_ix};
    if problem.dimension == 1
        figure(p_ix); clf;
        fprintf('Plotting %s...\n', problem.name );
        xrange = linspace(problem.prior.mean - 2*sqrt(problem.prior.covariance), ...
                          problem.prior.mean + 2*sqrt(problem.prior.covariance), ...
                          n)';
        h_prior = plot(xrange,...
            mvnpdf(xrange, problem.prior.mean, problem.prior.covariance), 'g--', 'LineWidth', 1); hold on;
        like_func_vals = ...
            exp(problem.log_likelihood_fn(...
            [xrange zeros(n, problem.dimension - 1)]));
        like_func_vals = like_func_vals ./ max(like_func_vals) ...
            .* mvnpdf(0, 0, problem.prior.covariance(1));
        
        h_ll = plot(xrange, like_func_vals, 'b', 'LineWidth', 1);
       % h_post = plot(xrange, like_func_vals ...
        %              .*mvnpdf(xrange, problem.prior.mean, problem.prior.covariance), ...
         %             'r', 'LineWidth', 1);
                  
       
        %legend([h_prior h_ll h_post], {'Prior', 'Likelihood', 'Posterior'});
        title(problem.name);
        
        xlim([xrange(1), xrange(end)]);
        % remove axes
        set(gca,'ytick',[]);
        set(gca,'xtick',[]);
        set(gca,'yticklabel',[]);
        set(gca,'xticklabel',[]);        
        
        set(gcf,'units','centimeters')
        %set(gcf,'Position',[1 1 4 4])
        %savepng(gcf, [plotdir problem.name] );
        
        filename = sprintf('%s%s', plotdir, strrep( problem.name, ' ', '_' ));
        %matlab2tikz( filename, 'height', '\fheight', 'width', '\fwidth', 'showInfo', false, 'showWarnings', false );
        %fprintf('\\input{%s}\n', filename);
        set_fig_units_cm( 2.5, 2.5 );
        matlabfrag(filename);        
    end
end
