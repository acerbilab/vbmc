function [dist,edge_mat,edge_list] = setdist(full_dist_mat,red,blue)%,red_nodes,blue_nodes,allnodes);
% dist: the distance between two sets.
% edge_mat: the specified edges, as returned in a square matrix.
% edge_list: the index-pairs corresponding to edges.
% dist_mat: the square matrix of distances between each pair of points in
% the union of the two sets. dist_mat need only be upper triangular.
% red: the (row & column) indices that correspond to points belonging to
% the first set.
% blue: the (row & column) indices that correspond to points belonging to
% the second set.

% start with all edges.

num_red = length(red);
red_and_blue = [red,blue];
[actual_used_edges,dummy_inds,inds] = unique(red_and_blue);
full_dist_mat = full_dist_mat(actual_used_edges,actual_used_edges);
dist_mat = triu(full_dist_mat);

red = inds(1:num_red);
blue = inds((num_red+1):end);


edge_mat = triu(ones(length(dist_mat)),1);

still_greedy = true;
first_time = true;

for ind = 1:length(edge_mat(:)) % dummy ind
    

    % If there are forks left, delete an edge from A fork (doesn't matter which one?). Otherwise, just
    % delete the longest that leavse things connected. 
    
    % only search over edges from a node of colour A to colour B if there
    % is another edge from that node to another node of colour B. That is,
    % only search over edges that are part of a fork. 
    
    if still_greedy
        greedy_edges = greedy_edges_fn(edge_mat,red,blue);
        
        max_greediness = max(greedy_edges(:));
        still_greedy = max_greediness>0;

        dist_of_maximally_greedy_edges = (max_greediness==greedy_edges).*dist_mat;

        [distys,edge2delete_1] = max(dist_of_maximally_greedy_edges);
        [disty,edge2delete_2] = max(distys);
        edge2delete = [edge2delete_1(edge2delete_2),edge2delete_2];
    else
        % Delete longest edge, but we have to start checking that we don't disconnect the graph
        
        if first_time
            first_time = false;
            actual_edges = find(edge_mat~=0);
            [distys,edge_inds] = sort(dist_mat(actual_edges),'descend');
            [xs,ys]=ind2sub(size(dist_mat),actual_edges(edge_inds));
        end
        
        edge2delete = zeros(0,2);
        while ~isempty(xs)
            temp_edge2delete = [xs(1),ys(1)];
            % Either we are going to actually delete this edge, or, if we
            % determine that deleting it would disconnect the graph, we can
            % remove it from further consideration.
            xs(1)= [];
            ys(1)= [];
            
            [temp_edge_mat] = delete_edge(edge_mat,temp_edge2delete);
        
            remains_connected = is_connected(temp_edge_mat,red,blue);
            if remains_connected
                edge2delete = temp_edge2delete;
                break
            end
        end
    end
    
    if ~isempty(edge2delete)
        edge_mat = delete_edge(edge_mat,edge2delete);
    else
        break
    end
  
%     if min_num_forks<5
%         xs = edge_list(:,1);
%         ys = edge_list(:,2);
% 
%         figure;
%         hold on;
%         plot(red_nodes(:,1),red_nodes(:,2),'ro','MarkerSize',15,'LineWidth',2);
%         plot(blue_nodes(:,1),blue_nodes(:,2),'bo','MarkerSize',15,'LineWidth',2);
% 
% 
%         dist1 = 0;
%         for i = 1:length(xs)
%             xind = xs(i);
%             yind = ys(i);
%             line([allnodes(xind,1),allnodes(yind,1)],[allnodes(xind,2),allnodes(yind,2)],'Color','k','LineWidth',2);
%             dist1 = dist1 + sqrt((allnodes(xind,1)-allnodes(yind,1))^2 + (allnodes(xind,2)-allnodes(yind,2))^2);
%         end
%         keyboard
%     end
    
end

temp_edge_mat = edge_mat + edge_mat';

red_nodes_connected_to_red_and_blue = red((and(sum(temp_edge_mat(red,red))==1 , sum(temp_edge_mat(blue,red))==1)));

for i = 1:length(red_nodes_connected_to_red_and_blue)
    node = red_nodes_connected_to_red_and_blue(i);
    red_node = red((temp_edge_mat(node,red)==1));
    blue_node = blue((temp_edge_mat(node,blue)==1));
    if full_dist_mat(node,blue_node)>full_dist_mat(blue_node,red_node)
        min_node = min([blue_node,red_node]);
        max_node = max([blue_node,red_node]);
        edge_mat(min_node,max_node) = 1;
        edge_mat(node,blue_node) = 0;
        edge_mat(blue_node,node) = 0;
    end
end

blue_nodes_connected_to_red_and_blue = blue((and(sum(temp_edge_mat(blue,blue))==1 , sum(temp_edge_mat(red,blue))==1)));

for i = 1:length(blue_nodes_connected_to_red_and_blue)
    node = blue_nodes_connected_to_red_and_blue(i);
    red_node = red((temp_edge_mat(node,red)==1));
    blue_node = blue((temp_edge_mat(node,blue)==1));
    if full_dist_mat(node,red_node)>full_dist_mat(blue_node,red_node)
        min_node = min([blue_node,red_node]);
        max_node = max([blue_node,red_node]);
        edge_mat(min_node,max_node) = 1;
        edge_mat(node,red_node) = 0;
        edge_mat(red_node,node) = 0;
    end
end


dist = sum(sum(edge_mat.*dist_mat));
if nargout>2
edge_list = edge_mat2list(edge_mat);
end

function edge_list = edge_mat2list(edge_mat)
edge_list = [];
[edge_list(:,1),edge_list(:,2)] = find(edge_mat);

function [edge_mat] = delete_edge(edge_mat,edge2delete)
    x = edge2delete(1);
    y = edge2delete(2);
    if x>y
        x = edge2delete(2);
        y = edge2delete(1);
    end

    edge_mat(x,y) = 0;

function ic=is_connected(edge_mat,red,blue) 
% true iff all red nodes are connected to a blue node and all blue nodes
% are connected to a red node.

% edge_list = edge_mat2list(edge_mat);
% 
% components = grComp(edge_list,max([red,blue]));
edge_mat=edge_mat+edge_mat';
[S,components] = graphconncomp(sparse(edge_mat));

num_components = max(components);
all_components = 1:num_components;

blue2red = isempty(setdiff(all_components,(components(red))));
red2blue = isempty(setdiff(all_components,(components(blue))));
ic = blue2red && red2blue;

function greedy_edges = greedy_edges_fn(edge_mat,red,blue)
% if a fork is higher degree, it counts as multiple forks

% edge_mat is currently just upper triangular
edge_mat = edge_mat + edge_mat';
RBmat = edge_mat(red,blue);

[num_red,num_blue] = size(RBmat);

edge_to_other_reds = (repmat(sum(RBmat),num_red,1)-RBmat)>0;
edge_to_other_blues = (repmat(sum(RBmat,2),1,num_blue)-RBmat)>0;

greedy_edges_RB = (edge_to_other_reds+edge_to_other_blues).*RBmat;

greedy_edges = 0*edge_mat; %faster than doing size(edge_mat)
greedy_edges(red,blue) = greedy_edges_RB;
greedy_edges(blue,red) = greedy_edges_RB';
greedy_edges = triu(greedy_edges);

% num_blue2red = sum(sum(RBmat(:,sum(RBmat)>1)));
% num_red2blue = sum(sum(BRmat(:,sum(BRmat)>1)));
% 
% n = num_red2blue + num_blue2red;