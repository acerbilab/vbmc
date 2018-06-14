function X_min = fast_exp_loss_min(exp_loss, ...
                lower_bound, upper_bound, exp_loss_evals, ...
                X_data, sqd_input_scales, y_min_ind, opt)
% find the position at which fn exp_loss is minimised.

if nargin<8
    opt.plots = false;
end

num_start_pts = 7;

bounds = [lower_bound;upper_bound];

if opt.derivative_observations
    plain_obs = find(X_data(:,end) == 0);
    X_data = X_data(plain_obs,1:end-1);
    exp_loss = @(x) exp_loss([x,0]);
    y_min_ind = find(plain_obs==y_min_ind);
end

num_data = size(X_data,1);
num_dims = size(X_data,2);


if num_data <= num_start_pts
    start_pt_inds = 1:num_data;
else
    start_pt_inds = round(linspace(rem(num_data-1, num_start_pts)+1,...
        num_data, num_start_pts-1));
    start_pt_inds = unique([y_min_ind, start_pt_inds]); 
end
num_start_pts = length(start_pt_inds);
start_pts = X_data(start_pt_inds,:);

num_line_pts = floor(0.5*exp_loss_evals/num_start_pts);

best_X = nan(num_start_pts, num_dims);
best_loss = nan(num_start_pts,1);

if opt.plots
    switch num_dims
        case 1
            x = linspace(lower_bound,upper_bound, 100);
            f = exp_loss(x);
            plot(x,f,'r');
            hold on
        case 2

            num = 50;

            x1 = linspace(lower_bound(1),upper_bound(1), num)';
            x2 = linspace(lower_bound(2),upper_bound(2), num)';
            x = allcombs([x1,x2]);
            f = exp_loss(x);

            X2 = repmat(x2,1,num);
            X1 = repmat(x1',num,1);
            F = reshape(f,num,num);

            max_y = max(f+1);

            figure(3)
            clf
            contourf(X1,X2,F);
            colorbar
            hold on
            plot3(X_data(:,1),X_data(:,2),max_y*ones(num_data,1),'w+','LineWidth',5,'MarkerSize',10)
            plot3(X_data(:,1),X_data(:,2),max_y*ones(num_data,1),'k+','LineWidth',3,'MarkerSize',9)

    end
end


% % assume the input scales for the expected loss surface are proportional to
% % those of the objective fn.
% [unused_max_logL, best_ind] = max([gp.hypersamples(:).logL]);
% input_scales = exp(...
%     gp.hypersamples(best_ind).hyperparameters(gp.input_scale_inds));



% We find the best expected loss local to a small number of starting points

dist_stack_start_data = ...
        abs(bsxfun(@minus,...
                reshape(start_pts,num_start_pts,1,num_dims),...
                reshape(X_data,1,num_data,num_dims)));
            
self_inds_mat = [repmat([(1:num_start_pts)',start_pt_inds'],num_dims,1), ...
                kron2d((1:num_dims)', ones(num_start_pts,1))];
self_inds = sub2ind(size(dist_stack_start_data),...
            self_inds_mat(:,1), self_inds_mat(:,2), self_inds_mat(:,3));
dist_stack_start_data(self_inds) = nan;
            
dist_mat = sqrt(...
    sum(...
    bsxfun(@rdivide, ... 
    dist_stack_start_data.^2, ...
    reshape(sqd_input_scales, 1, 1, num_dims)), ...
    3));
closest_dist = min(dist_mat,[],2);

input_scales = bsxfun(@times, max(1,closest_dist), sqrt(sqd_input_scales));

[unused_f,g] = exp_loss(start_pts);
zoomed = simple_zoom_pt(start_pts, g, input_scales, 'minimise');
x = cap(zoomed, lower_bound, upper_bound);
f = exp_loss(x);

% Then, for each starting point, we perform a line search in the direction
% given by the difference between the starting point and its local optimum.
% More accurately, for a number of points along that line, we perform
% another local optimisation.

%par
for start_pt_ind = 1:num_start_pts
    
    start_pt = start_pts(start_pt_ind,:);
    
    best_X_line = nan(num_line_pts, num_dims);
    best_loss_line = nan(num_line_pts,1);
    
    best_X_line(1,:) = x(start_pt_ind,:);
    best_loss_line(1) = f(start_pt_ind,:);
    
    direction = x(start_pt_ind,:) - start_pt;
    
    exploration_time = norm(direction) == 0;
    if exploration_time
        % move towards a corner. The corner is chosen by rotating through
        % them all using a binary conversion.
        
        bin_vec = de2bi(start_pt_ind+start_pt_inds(end))+1;
        
        sel_vec = ones(1,num_dims);
        num = min(length(sel_vec),length(bin_vec));
        sel_vec(1:num) = bin_vec(1:num);
        
        direction =...
            bounds(sub2ind([2,num_dims],sel_vec,1:num_dims)) - start_pt;
        if norm(direction) == 0
            direction = mean([lower_bound;upper_bound]) - start_pt;
        end
    end
%     unzeroed_direction = direction;
%     unzeroed_direction(abs(direction)<eps) = eps;

    ups = (upper_bound - start_pt)./direction;
    downs = (lower_bound - start_pt)./direction;
    bnds = sort([ups;downs]);
    min_bnd = max(bnds(1,:));
    max_bnd = min(bnds(2,:));
    
    if exploration_time
        min_further_away = abs(min_bnd) > abs(max_bnd);
        if min_further_away
            line_pts = linspace(0, min_bnd, num_line_pts)';
        else
            line_pts = linspace(0, max_bnd, num_line_pts)';
        end
    else
        line_pts = linspace(0, max_bnd, num_line_pts)';
    end
    % no point evaluating at start_pt
    line_pts = line_pts(2:end);

    X_line = @(line_pt) bsxfun(@plus, start_pt, ...
        bsxfun(@times, line_pt, direction));
    
%     if opt.plots
%         
%         switch num_dims
%             case 1
%                 
%             case 2
%                 coords = [X_line(min_bnd); X_line(max_bnd)];
%                 line(coords(:,1), coords(:,2), [max_y,max_y], 'Color','w')
%         end
%     end


    %parfor line_pt_ind = 2:num_line_pts
%         line_pt = line_pts(line_pt_ind-1);
%         X_line_pt = X_line(line_pt);

        X_line_pts = X_line(line_pts);
        [original_loss, g] = exp_loss(X_line_pts);
        zoomed = ...
            simple_zoom_pt(X_line_pts, g, input_scales(start_pt_ind,:), ...
            'minimise');
        zoomed = cap(zoomed, lower_bound, upper_bound);
        
        zoomed_loss = exp_loss(zoomed);
        

        
        improvement = original_loss > zoomed_loss;
        % the first point is the original starting point
        improved_inds = find(improvement)+1;
        not_improved_inds = find(~improvement)+1;
        
        
        best_X_line(improved_inds, :) = zoomed(improvement, :);
        best_loss_line(improved_inds) = zoomed_loss(improvement);
        best_X_line(not_improved_inds, :) = X_line_pts(~improvement, :);
        best_loss_line(not_improved_inds) = original_loss(~improvement);
        
%         if opt.plots
%             switch num_dims 
%                 case 1
%                     plot(zoomed,best_loss_line(line_pt_ind),'ro','MarkerSize',6)
%                 case 2
%                     plot(X_line_pts(:,1),X_line_pts(:,2),'k.','MarkerSize',10)
%                     plot(zoomed(:,1),zoomed(:,2),'k.','MarkerSize',14)
%                     plot3(zoomed(1),zoomed(2),max_y,'w.','MarkerSize',6)
%             end
%         end
%    end
    
    % do a flipud so, if there is a tie, we return a point closer to the
    % corners than to the start point.
    [min_best_loss_line, min_ind_line] = min(flipud(best_loss_line));
    min_ind_line = num_line_pts + 1 - min_ind_line;

    best_X(start_pt_ind,:) = best_X_line(min_ind_line,:);
    best_loss(start_pt_ind,:) = min_best_loss_line;
end

[min_best_loss, min_ind] = min(best_loss);
X_min = best_X(min_ind,:);

if opt.plots
    switch num_dims
        case 1
            plot(X_min,min_best_loss,'r+','MarkerSize',10)
            refresh
            drawnow
        case 2
            plot3(X_min(1),X_min(2),max_y,'w+','LineWidth',5,'MarkerSize',10)
            refresh
            drawnow
    end
end

function x = cap(x, lower_bound, upper_bound)
x = bsxfun(@min, bsxfun(@max, x, lower_bound), upper_bound);