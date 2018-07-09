function [XData,filter,distance_matrix] = remove_close_data(XData,scales,closeness_num)
% downsample such that the data is well-separated, removing all samples
% that are close than closeness_num units, using the Mahalanobis distance
% with the diagonal covariance matrix equal to diag(scales.^2).

if nargin<3
closeness_num = 1;
end
[NData,NDims] = size(XData);

% Commented code is slower and I think slightly wrong
% if NDims==1 && nargout<3
%     [sorted_XData, sort_order] = sort(XData);
%     
%     mat = repmat([0;diff(sorted_XData)/scales],1,NData);
%     mat(triu(true(NData)))=0;
%     
%     distance_matrix = cumsum(mat)';
%     [problem_xinds,problem_yinds]=poorly_conditioned(distance_matrix,closeness_num); 
%     
%     problem_xinds = find(diff(sorted_XData)/scales < closeness_num);
%     problem_yinds = problem_xinds+1;
%     
%     problem_xinds = sort_order(problem_xinds);
%     problem_yinds = sort_order(problem_yinds);
%     
% else
    distance_matrix = sqrt(squared_distance(XData,XData,scales));
    [problem_xinds,problem_yinds]=poorly_conditioned(distance_matrix,closeness_num); 
%end

dropped=[];
while ~isempty(problem_xinds) % We still have problems

    % Remove the sample with the lowest logL - if there's a tie,
    % remove the sample that lead to the most problems
    preference_matrix=histc([problem_xinds;problem_yinds],1:NData); % note the negative sign here
    preference_matrix(preference_matrix(:,1)<1,2)=-inf; % don't drop samples that do not give us problems
    [sorted,priority_order]=sortrows(preference_matrix); %sorts ascending

    drop=priority_order(end);
    dropped=[dropped,drop];

    % The indices of the problems that will be solved by dropping this
    % point.
    problem_drop=unique([find(problem_xinds==drop);find(problem_yinds==drop)]);

    problem_xinds(problem_drop)=[];
    problem_yinds(problem_drop)=[];
end
filter = 1:NData;
filter = setdiff(filter,dropped);

XData = XData(filter,:);

function [xinds,yinds]=poorly_conditioned(scaled_distance_matrix,num)
% closer than num scales is too close for comfort

Nsamples=size(scaled_distance_matrix,1);
scaled_distance_matrix(tril(true(Nsamples)))=inf;
[xinds,yinds]=find(scaled_distance_matrix<num);
    