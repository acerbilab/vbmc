function [index_vals,vector_vals,final_node] = kd_closestpointgood(tree,point,node_number)


% pramod vemulapalli 02/08/2010
% isnpired by work done by Andrea Tagliasacchi, simon fraiser university
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% INPUTS
% tree        --- the cell array that contains the tree
% point       --- the point of interest
% none_number --- Internal Variable, Donot Define

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% OUTPUTS
% index_vals  --- the index value in the original matrix that was used to
%                 build the tree
% vector_vals --- the feature vector closest to the given vector
% final_node  --- the node that contains the closest vector

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Donot define the node_number variable -- it is used for internal
% referencing


global tree_cell_2;
global best_node;
global best_dist;
global safety_check_2;

% in the first iteration .... setup everything
if (nargin==2)

    safety_check_2=0;
    [index_vals,vector_vals,final_node] = kd_closestpointfast(tree,point);
    node_number=1;
    tree_cell_2=tree;
    clear tree;

    best_node=final_node;
    best_dist=sqrt(sum((tree_cell_2(final_node).nodevector-point).^2));

end

if (isempty(safety_check_2))
    error ('Too Many input variables .. you only need to pass the tree and the point of interest (2 arguments)');
end



% if leaf check if distance is smaller than best distance
if(strcmp(tree_cell_2(node_number).type,'leaf'))
    sqrd_dist=sqrt(sum((tree_cell_2(node_number).nodevector-point).^2));
    if (sqrd_dist<best_dist)
        best_dist=sqrd_dist;
        best_node=node_number;
    end
    return;
end


% distance to the segmenting hyper surface
dist_current_node=( point(tree_cell_2(node_number).splitdim) - (tree_cell_2(node_number).nodevector(tree_cell_2(node_number).splitdim)) );


% if hypersphere radius is less than distance to the hyper surface
if (best_dist<abs(dist_current_node))

    % choose left if the point is to the left of hypersurface
    if (dist_current_node<0)

        if (~isempty(tree_cell_2(node_number).left))
            kd_closestpointgood(0,point,tree_cell_2(node_number).left);
        else
            return;
        end

    else

        % otherwise choose right
        if (~isempty(tree_cell_2(node_number).right))
            kd_closestpointgood(0,point,tree_cell_2(node_number).right);
        else
            return;
        end


    end
else


    % as the hypersphere radius is greater than the distance to the hyper
    % surface.. check both sides

    % check to see if the node is  a good point
    sqrd_dist=sqrt(sum((tree_cell_2(node_number).nodevector-point).^2));
    if (sqrd_dist<best_dist)
        best_dist=sqrd_dist;
        best_node=node_number;
    end


    % check to see if any points to the left of the node are good
    if (~isempty(tree_cell_2(node_number).left))
        kd_closestpointgood(0,point,tree_cell_2(node_number).left);
    else
        return;
    end


    % check to see if any points to the right of the node are good
    if (~isempty(tree_cell_2(node_number).right))
        kd_closestpointgood(0,point,tree_cell_2(node_number).right);
    else
        return;
    end    
    

end

% output the correct values and clear up all the other variables
if (nargin==2)
    index_vals=tree_cell_2(best_node).index;
    vector_vals=tree_cell_2(best_node).nodevector;
    final_node=best_node;
    clear global tree_cell_2;
    clear global best_node;
    clear global best_dist;
    clear global safety_check_2;
end













