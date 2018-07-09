function [index_vals,dist_vals,vector_vals] = kd_nclosestpoints(tree,point,n,first_iter_flag)


% the code gives you a really fast, approximate value of the closest point

% tree is the kd tree
% point is the query point for which we would want to find the nearest
% point in the kd tree
% first_iter_flag - please donot define this variable; this is an internal
% variable that is used for recursion 
% pramod vemulapalli 02/07/2010 

% index_vals are the closest points
if(nargin<3)
    error('Not enough input arguments ...');
end 


% in the first iteration make sure that the data is in the right shape 
if(nargin==3)
    if(n>tree.numpoints)
        error('Not enough points in the tree ...');
    end
    size_point=size(point);
    if (size_point(1)>size_point(2))
        point=point';  % transpose the point data if it is given as a single column instead of a single row
    end
    first_iter_flag=0;
end


% if a leaf then do the calculating nicely 
if(strcmp(tree.type,'leaf'))
    index_vals=tree.index;
    dist_vals=sqrt(sum((tree.centroid-point).^2));
    vector_vals=tree.centroid;
    return;
end

if (point(tree.splitdim)<tree.splitval)
    
    if (tree.left.numpoints<n) % check to see if there are enough points in the branches ahead 
        [index_vals1,dist_vals1,vector_vals1]=kd_nclosestpoints(tree.left,point,n,first_iter_flag);  
        [index_vals2,dist_vals2,vector_vals2]=kd_nclosestpoints(tree.right,point,n,first_iter_flag);
        index_vals=[index_vals1;index_vals2];
        dist_vals=[dist_vals1;dist_vals2];
        vector_vals=[vector_vals1;vector_vals2];
    else
        [index_vals,dist_vals,vector_vals]=kd_nclosestpoints(tree.left,point,n,first_iter_flag);
    end
    
else
    
    if (tree.right.numpoints<n)
        [index_vals1,dist_vals1,vector_vals1]=kd_nclosestpoints(tree.left,point,n,first_iter_flag);  
        [index_vals2,dist_vals2,vector_vals2]=kd_nclosestpoints(tree.right,point,n,first_iter_flag);
        index_vals=[index_vals1;index_vals2];
        dist_vals=[dist_vals1;dist_vals2];
        vector_vals=[vector_vals1;vector_vals2];
    else
        [index_vals,dist_vals,vector_vals]=kd_nclosestpoints(tree.right,point,n,first_iter_flag);
    end
    
end


% incase if this is the first iteration 
% pack everything neatly and send it out 
if(nargin==3)
      final_mat=[index_vals,dist_vals,vector_vals];
      rearranged_final_mat = sortrows(final_mat,2);
      index_vals=rearranged_final_mat(1:n,1);
      dist_vals=rearranged_final_mat(1:n,2);
      vector_vals=rearranged_final_mat(1:n,3:end);
end 