function [index_vals,vector_vals,final_nodes] = kd_knn(tree,point,k,plot_stuff,node_number)


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
% referencin

global best_points_mat
global number_of_points
global tree_cell
global safety_check;




dim =size(point,2);
index_vals=0;
vector_vals=0;
final_nodes=0;
debug_val=plot_stuff;

if(debug_val); global h; end

if (nargin==4)
    safety_check=0;
    best_points_mat=zeros(k,dim+1+1+1);% for index in tree, index in original mat, dist to point
    number_of_points=0;
    node_number=1;

    if(debug_val); h = plot(best_points_mat(1:k,1),best_points_mat(1:k,2),'g*'); end;
    tree_cell=tree;
    clear tree;
end

if (isempty(safety_check))
    error ('Insufficient number of input variables ... please check ');
end


% if the current node is a leaf then output its results
if(strcmp(tree_cell(node_number).type,'leaf'))
    node_check(point,k,node_number,debug_val);
    return;
end

% if the current node is not a leaf
% check to see if the point is to the left of the split dimension
% if it is to the left then recurse to the left
if (point(tree_cell(node_number).splitdim)<=tree_cell(node_number).splitval)
    if (isempty(tree_cell(node_number).left))
        % incase the left node is empty, then output current results
    else
        kd_knn(0,point,k,plot_stuff,tree_cell(node_number).left);
    end
else
    % as the point is to the right of the split dimension
    % recurse to the right
    if (isempty(tree_cell(node_number).right))
        % incase the right node is empty, then output current results
    else
        kd_knn(0,point,k,plot_stuff,tree_cell(node_number).right);
    end    
end




% do the computation to decide if you need to search on the otherside
if (number_of_points<k)
    search_otherside= 'true';
else

    sum_value=0;
    dist_of_kthpoint=best_points_mat(k,dim+3);

    for i=1:dim

        if (point(i)<tree_cell(node_number).hyperrect(1,i))
            sum_value = sum_value + (point(i)-tree_cell(node_number).hyperrect(1,i)).^2;
            if (sum_value > dist_of_kthpoint)
                search_otherside='false';
                break;
            end
        elseif (point(i)>tree_cell(node_number).hyperrect(2,i))
            sum_value = sum_value + (point(i)-tree_cell(node_number).hyperrect(2,i)).^2;
            if (sum_value > dist_of_kthpoint)
                search_otherside='false';
                break;
            end
        end
    end

    search_otherside= 'true';
end



if (strcmp(search_otherside,'true'))

    node_check(point,k,node_number,debug_val);

    % if the current node is not a leaf
    % check to see if the point is to the left of the split dimension
    % if it is to the left then decide whether to recurse to the right
    if (point(tree_cell(node_number).splitdim)<=tree_cell(node_number).splitval)
        if (isempty(tree_cell(node_number).right))
            % incase the right node is empty, then output current results

        else
            % as the point is to the right of the split dimension
            % decide whether to recurse to the left
            kd_knn(0,point,k,plot_stuff,tree_cell(node_number).right);
        end  
    else
        if (isempty(tree_cell(node_number).left))
            % incase the left node is empty, then output current results

        else
            % as the point is to the right of the split dimension
            % decide whether to recurse to the left
            kd_knn(0,point,k,plot_stuff,tree_cell(node_number).left);
        end
    end

end

if (nargin==4)

    vector_vals=best_points_mat(1:number_of_points,1:dim);
    index_vals=best_points_mat(1:number_of_points,dim+1);
    final_nodes=best_points_mat(1:number_of_points,dim+2);

    clear global best_points_mat;
    clear global number_of_points;
    clear global tree_cell;
    clear global safety_check;

end


function []=node_check(point,k,node_number,debug_val)

global best_points_mat
global number_of_points
global tree_cell
if(debug_val); global h; end

dim =size(point,2);
distance = sum((point-tree_cell(node_number).nodevector).^2);

if (number_of_points==k && best_points_mat(k,dim+3)>distance)
    best_points_mat(k,1:dim)=tree_cell(node_number).nodevector;
    best_points_mat(k,dim+1)=tree_cell(node_number).index;
    best_points_mat(k,dim+2)=node_number;
    best_points_mat(k,dim+3)=distance;
    best_points_mat=sortrows(best_points_mat,dim+3);
    if(debug_val);
        set(h,'XData',best_points_mat(1:k,1))
        set(h,'YData',best_points_mat(1:k,2))
    end
elseif(number_of_points<k)
    number_of_points=number_of_points+1;
    best_points_mat(number_of_points,1:dim)=tree_cell(node_number).nodevector;
    best_points_mat(number_of_points,dim+1)=tree_cell(node_number).index;
    best_points_mat(number_of_points,dim+2)=node_number;
    best_points_mat(number_of_points,dim+3)=distance;
    % once the variable gets filled up then sort the rows 
    if(number_of_points==k)
        best_points_mat=sortrows(best_points_mat,dim+3);
    end
    if(debug_val);
        set(h,'XData',best_points_mat(1:k,1))
        set(h,'YData',best_points_mat(1:k,2))
    end
end

return;
