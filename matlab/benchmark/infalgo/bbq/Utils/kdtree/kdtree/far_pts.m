function pts = far_pts(X, box, num_pts)
% pts = far_pts(X, lower_bnd, upper_bnd, num_pts)
% use a kd-tree to return the (num_pts) points farthest from X within the
% box. This is an approximate solution to the largest empty spheres
% problems, better solved, slower, by the voronoi diagram (see
% find_farthest.m)

[num, dim] = size(X);

lower_bnd = box(1,:);
upper_bnd = box(2,:);

if nargin<3
    num_pts = num;
end

corners = allcombs([lower_bnd;upper_bnd]);
X=[X;corners];

if dim == 1
    pts = find_farthest_1d(X, box, num_pts);
else
    tree = kd_buildtree(X,0);
    inds = [tree.numpoints]~=1;%1:length(tree);%

    means = arrayfun(@(x) mean(x.hyperrect), tree(inds), 'UniformOutput', false);
    nodes = arrayfun(@(x) x.nodevector, tree(inds), 'UniformOutput', false);

    pts = vertcat(means{:});
    node_pts = vertcat(nodes{:});

    dists = (pts-node_pts).^2*ones(dim,1);
    [sorted_dists, sort_inds] = sort(dists,1,'descend');

    num_pts = min(num_pts, length(sort_inds));

    pts = pts(sort_inds(1:num_pts),:);
end