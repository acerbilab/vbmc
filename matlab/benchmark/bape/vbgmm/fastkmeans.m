function [idx,C,mindist,q2,quality] = fastkmeans(X,k,options)
%FASTKMEANS Fast K-means clustering.
%   IDX = FASTKMEANS(X,K) partitions the points in the N-by-D data matrix X
%   into K clusters.  This partition minimizes the sum, over all clusters, 
%   of the within-cluster sums of point-to-cluster-centroid distances.
%   Rows of X correspond to points, columns correspond to variables. 
%   FASTKMEANS uses squared Euclidean distances. IDX is the index, for each
%   data point, of the cluster to which it belongs.
%
%   If K is a scalar, it is taken to be the number of clusters desired. If
%   K is a M-by-D array, it is taken to be the D-dimensional coordinates of
%   M points, to be used as initial guesses for the cluster centers.
%
%   FASTKMEANS treats NaNs as missing data, and ignores any rows of X that
%   contain NaNs.
%
%   IDX = FASTKMEANS(X,K) partitions the points in the N-by-D data matrix X
%   into K clusters.  This partition minimizes the sum, over all clusters, 
%   of the within-cluster sums of point-to-cluster-centroid distances.
%   Rows of X correspond to points, columns correspond to variables. 
%   FASTKMEANS uses squared Euclidean distances. IDX is the index, for each
%   data point, of the cluster to which it belongs.
%   
%   IDX = FASTKMEANS(X,K,OPTIONS) replaces the default algorithm parameters
%   with values in the structure OPTIONS. FASTKMEANS uses these options:
%
%      OPTIONS.Display defines the level of display. Accepted values for
%      Display are 'iter', 'notify', 'final', and 'off' for no display. The 
%      default value of Display is 'off'. 
%
%      OPTIONS.Method selects the algorithm to be used.
%      * 0, unoptimized, using n by k matrix of distances O(nk) space;
%      * 1, vectorized, using only O(n+k) space;
%      * 2, like 1, in addition using distance inequalities (default).
%
%      OPTIONS.Preprocessing specifies the preprocessing step performed on
%      the raw data matrix X.
%      * 'none' performs no preprocessing (default);
%      * 'normalize' normalizes the data to have zero mean and unit 
%        variance along each coordinate axis;
%      * 'whiten' normalizes the data to have zero mean and identity
%        covariance matrix.
%
%   [IDX,C] = FASTKMEANS(...) returns the K cluster centroid locations in
%   the K-by-D matrix C.
%
%   [IDX,C,MINDIST] = FASTKMEANS(...) returns an upper bound MINDIST of the
%   distance of each point to the nearest center. The distance is returned
%   in the transformed coordinates (after whitening or normalization).
%
%   [IDX,C,MINDIST,Q2] = FASTKMEANS(...) returns the mean of UDIST^2 in
%   transformed coordinates (after whitening or normalization).
%
%   [IDX,C,MINDIST,Q2,QUALITY] = FASTKMEANS(...) returns the mean of UDIST
%   in transformed coordinates (after whitening or normalization).
%

%   Author: Charles Elkan
%   Interface and options: Luigi Acerbi
%   Email:  luigi.acerbi@gmail.com
% 
%  Reference:
%  Charles Elkan, Using the Triangle Inequality to Accelerate k-Means,
%  Proceedings of the Twentieth International Conference on Machine Learning 
%  (ICML-2003), Washington DC, 2003.

if nargin < 3; options = []; end

% Remove NaNs
nanidx = any(isnan(X),2);
X(nanidx,:) = [];

[n,dim] = size(X);

% Default options
defopts.Display         = 'off';        % Display
defopts.Method          = 2;            % Clustering method
defopts.Preprocessing   = 'none';       % Preprocessing step

% Assign default options if not defined
for f = fields(defopts)'
    if ~isfield(options,f{:}) || isempty(options.(f{:}))
        options.(f{:}) = defopts.(f{:});
    end
end

switch options.Display
    case {'notify','notify-detailed'}
        trace = 2;
    case {'none', 'off'}
        trace = 0;
    case {'iter','iter-detailed'}
        trace = 3;
    case {'final','final-detailed'}
        trace = 1;
    otherwise
        trace = 1;
end

method = options.Method;

switch lower(options.Preprocessing)
    case 'normalize'
        mu = mean(X,1);
        X = bsxfun(@minus,X,mu);
        sigma = std(X,[],1);
        X = bsxfun(@rdivide,X,sigma);        
    case 'none'
        % Do nothing        
    case 'whiten'
        mu = mean(X,1);
        X = bsxfun(@minus,X,mu);
        M = X'*X / n;
        [U,S] = svd(M);
        X = X*U'*diag(1./sqrt(diag(S+eps)));
    otherwise
        error('Unknown preprocessing method in OPTIONS.Preprocessing. Available methods are ''none'',''normalize'' and ''whiten''.');
end

if isscalar(k)
    [C,idx,mindist,lowr,computed] = anchors(mean(X),k,X);
    total = computed;
    skipestep = 1;
else 
    C = k;
    idx = zeros(n,1);
    total = 0;
    skipestep = 0;
    [k,dim2] = size(C);    
    if dim ~= dim2 error('dim(data) ~= dim(centers)'); end;
end

nchanged = n;
iteration = 0;
oldmincenter = zeros(n,1);

while nchanged > 0
    % do one E step, then one M step
    computed = 0;
    
    if method == 0 & ~skipestep
        for i = 1:n
            for j = 1:k
                distmat(i,j) = calcdist(X(i,:),C(j,:));
            end
        end
        [mindist,idx] = min(distmat,[],2);
        computed = k*n;

    elseif (method == 1 | (method == 2 & iteration == 0)) & ~skipestep
        mindist = Inf*ones(n,1);
        lowr = zeros(n,k);
        for j = 1:k
           jdist = calcdist(X,C(j,:));
           lowr(:,j) = jdist;
           track = find(jdist < mindist);
           mindist(track) = jdist(track);
           idx(track) = j;
        end
        computed = k*n;

    elseif method == 2 & ~skipestep 
        computed = 0;
%
% for each center, nndist is half the distance to the nearest center
% if d(x,center) < nndist then x cannot belong to any other center
% mindist is an upper bound on the distance of each point to its nearest center
%
        nndist = min(centdist,[],2);
% the following usually is not faster        
%        ldist = min(lower,[],2);
%        mobile = find(mindist > max(nndist(mincenter),ldist));
        mobile = find(mindist > nndist(idx));
        
% recompute distances for point i and center j 
%       only if j can possibly be the new nearest center
% for speed, the first check has been optimized by modifying centdist
% swapping the order of the checks is slower for data with natural clusters

        mdm = mindist(mobile);
        mcm = idx(mobile);
 
        for j = 1:k
% the following is incorrect: for j = unique(mcm)'
            track = find(mdm > centdist(mcm,j));
            if isempty(track) continue; end
            alt = find(mdm(track) > lowr(mobile(track),j));          
            if isempty(alt) continue; end
            track1 = mobile(track(alt));
%
% calculate exact distances to the mincenter
% recalculate separately for each jj to avoid copying too much of data
% redo may be empty, but we don't need to check this.
%
            redo = find(~recalculated(track1));
            redo = track1(redo);
            c = idx(redo);
            computed = computed + size(redo,1);
            for jj = unique(c)'
                rp = redo(find(c == jj));
                udist = calcdist(X(rp,:),C(jj,:));
                lowr(rp,jj) = udist;
                mindist(rp) = udist;
            end
            recalculated(redo) = 1;
            
            track2 = find(mindist(track1) > centdist(idx(track1),j));
            track1 = track1(track2);
            if isempty(track1) continue; end
           
            % calculate exact distances to center j
            track4 = find(lowr(track1,j) < mindist(track1));
            if isempty(track4) continue; end
            track5 = track1(track4);
            jdist = calcdist(X(track5,:),C(j,:));
            computed = computed + size(track5,1);
            lowr(track5,j) = jdist;
                    
            % find which points really are assigned to center j
            track2 = find(jdist < mindist(track5));
            track3 = track5(track2);
            mindist(track3) = jdist(track2);
            idx(track3) = j;
        end % for j=1:k
    end % if method
      
    oldcenters = C;
%       
% M step: recalculate the means for each cluster
% if a cluster is empty, its mean is left unchanged
% we minimize computations for clusters with little changed membership
%   
    diff = find(idx ~= oldmincenter);
    diffj = unique([idx(diff);oldmincenter(diff)])';
    diffj = diffj(find(diffj > 0));
    
    if size(diff,1) < n/3 & iteration > 0
         for j = diffj
            pls = find(idx(diff) == j);
            mins = find(oldmincenter(diff) == j);
            oldpop = pop(j);
            pop(j) = pop(j) + size(pls,1) - size(mins,1);
            if pop(j) == 0 continue; end
            C(j,:) = (C(j,:)*oldpop + sum(X(diff(pls),:),1) - sum(X(diff(mins),:),1))/pop(j); 
        end
    else
        for j = diffj
            track = find(idx == j);
            pop(j) = size(track,1);
            if pop(j) == 0 continue; end
% it's correct to have mean(X(track,:),1) but this can make answer worse!
            C(j,:) = mean(X(track,:),1);
        end
    end
    
    if method == 2
        for j = diffj
            offset = calcdist(C(j,:),oldcenters(j,:));
            computed = computed + 1;
            if offset == 0 continue; end
            track = find(idx == j);
            mindist(track) = mindist(track) + offset;
            lowr(:,j) = max(lowr(:,j) - offset,0);
        end
%
% compute distance between each pair of centers
% modify centdist to make "find" using it faster.
%
        recalculated = zeros(n,1);
        realdist = alldist(C);
        centdist = 0.5*realdist + diag(Inf*ones(k,1));
        computed = computed + k + k*(k-1)/2;   
    end
    
    nchanged = size(diff,1) + skipestep;
    iteration = iteration+1;
    skipestep = 0;
    oldmincenter = idx;

    if trace > 1
        fprintf ( 1, '%4d  %g  %d  %d\n', iteration, toc, nchanged, computed );
    end    
    total = total + computed;
end % while nchanged > 0

  udist = calcdist(X,C(idx,:));
  quality = mean(udist);
  q2 = mean(udist.^2);
  if trace > 0
      fprintf ( 1, '  %4d  %g  %g  %g  %d\n', iteration, toc, quality, q2, total );
      fprintf ( 1, '\n' );
      fprintf ( 1, 'KMEANS_FAST\n' );
      fprintf ( 1, '  Normal end of execution.\n' );
  end

% Account for NaNs in original data matrix
if any(nanidx)
    temp = NaN(n + sum(nanidx),1);
    temp(~nanidx,:) = idx;
    idx = temp;
    if nargout > 2
        temp = NaN(n + sum(nanidx),1);
        temp(~nanidx,:) = mindist;
        mindist = temp;
    end
end
  
% Return centroids in original space
if nargout > 1
    switch lower(options.Preprocessing)
        case 'normalize'
            C = bsxfun(@plus,bsxfun(@times,C,sigma),mu);
        case 'none'
            % Do nothing        
        case 'whiten'
            C = bsxfun(@plus,C*U'*diag(sqrt(diag(S+eps))),mu);
    end
end
  
end



%--------------------------------------------------------------------------
function centdist = alldist(centers)

% output: matrix of all pairwise distances
% input: data points (centers)

  k = size(centers,1);
  centdist = zeros(k,k);
  for j = 1:k
    centdist(1:j-1,j) = calcdist(centers(1:j-1,:),centers(j,:));
  end
  centdist = centdist+centdist';

end

%--------------------------------------------------------------------------
function [centers,mincenter,mindist,lowr,computed] = anchors(firstcenter,k,data)
% choose k centers by the furthest-first method

[n,dim] = size(data);
centers = zeros(k,dim);
lowr = zeros(n,k);
mindist = Inf*ones(n,1);
mincenter = ones(n,1);
computed = 0;
centdist = zeros(k,k);

for j = 1:k
    if j == 1
        newcenter = firstcenter;
    else
        [maxradius,i] = max(mindist);
        newcenter = data(i,:);
    end

    centers(j,:) = newcenter;
    centdist(1:j-1,j) = calcdist(centers(1:j-1,:),newcenter);
    centdist(j,1:j-1) = centdist(1:j-1,j)';
    computed = computed + j-1;
    
    inplay = find(mindist > centdist(mincenter,j)/2);
    newdist = calcdist(data(inplay,:),newcenter);
    computed = computed + size(inplay,1);
    lowr(inplay,j) = newdist;
                
    move = find(newdist < mindist(inplay));
    shift = inplay(move);
    mincenter(shift) = j;
    mindist(shift) = newdist(move);
end

end

%--------------------------------------------------------------------------
function distances = calcdist(data,center)
%  input: vector of data points, single center or multiple centers
% output: vector of distances

n = size(data,1);
n2 = size(center,1);

if n2 == 1
    distances = sum(data.^2, 2) - 2*data*center' + center*center';
elseif n2 == n
    distances = sum( (data - center).^2 ,2);
else
    error('Bad number of centers.');
end

% Euclidean 2-norm distance:
distances = sqrt(distances);

end