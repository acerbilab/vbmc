function [index_vals,dist_vals,vector_vals] = kd_rangequery(tree,point,range,node_number)

% pramod vemulapalli 02/07/2010
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% INPUTS
% tree        --- the cell array that contains the tree 
% point       --- the point of interest 
% range       --- +/- distance around each dimension of the point 
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



% check for sufficient number of inputs
if(nargin<3)
    error('Not enough input arguments ...');
end

global tree_cell;
global safety_check;

% in the first iteration make sure that the data is in the right shape
if(nargin==3)

    safety_check=0;
    node_number=1;
    tree_cell=tree;
    clear tree;
    
    dim=size(tree_cell(1).nodevector,2);
    
    size_range=size(range);
    size_point=size(point);
    
    % transpose the point data if it is given as a single column instead of
    % a single row
    if (size_point(1)>size_point(2))
        point=point';  
    end

    if ~(size_range(2)==dim && size_range(1)==2 && (sum(range(1,:)<=range(2,:))==dim) )
        error('range input not in correct format ...');
    end

end

if (isempty(safety_check))
        error ('Insufficient number of input variables ... please check ');
end 

% find dimension of feature vector 
dim=size(tree_cell(1).nodevector,2);

% if the current node is with in the range ... then store the data in the output  
if (sum(tree_cell(node_number).nodevector>=point+range(1,:))==dim && ...
        sum(tree_cell(node_number).nodevector<=point+range(2,:))==dim)
    
    index_vals=tree_cell(node_number).index;
    dist_vals=sqrt(sum((tree_cell(node_number).nodevector-point).^2));
    vector_vals=tree_cell(node_number).nodevector;

else
    
    index_vals=[];
    dist_vals=[];
    vector_vals=[];

end

% if the current node is a leaf then return 
if(strcmp(tree_cell(node_number).type,'leaf'))
    return;
end


% if the current node is not a leaf 
% check to see if the range hypercuboid is to the left of the split 
% and in that case send the left node out for inquiry 
if ( ((point(tree_cell(node_number).splitdim)+range(1,tree_cell(node_number).splitdim))<=tree_cell(node_number).splitval) && ...
        ((point(tree_cell(node_number).splitdim)+range(2,tree_cell(node_number).splitdim))<=tree_cell(node_number).splitval) )

    if (~isempty(tree_cell(node_number).left))
        [index_vals1,dist_vals1,vector_vals1]=kd_rangequery(0,point,range,tree_cell(node_number).left);
    else
        index_vals1=[];dist_vals1=[];vector_vals1=[];
    end
    index_vals=[index_vals;index_vals1];
    dist_vals=[dist_vals;dist_vals1];
    vector_vals=[vector_vals;vector_vals1];
    
end

% if the current node is not a leaf 
% check to see if the range hypercuboid is to the right of the split 
% and in that case send the right node out for inquiry 
if ( ((point(tree_cell(node_number).splitdim)+range(1,tree_cell(node_number).splitdim))>tree_cell(node_number).splitval) &&...
        ((point(tree_cell(node_number).splitdim)+range(2,tree_cell(node_number).splitdim))>tree_cell(node_number).splitval) )

    if (~isempty(tree_cell(node_number).left))
        [index_vals1,dist_vals1,vector_vals1]=kd_rangequery(0,point,range,tree_cell(node_number).right);
    else
        index_vals1=[];dist_vals1=[];vector_vals1=[];
    end
    index_vals=[index_vals;index_vals1];
    dist_vals=[dist_vals;dist_vals1];
    vector_vals=[vector_vals;vector_vals1];
    
end


% if the current node is not a leaf 
% check to see if the range hypercuboid stretches from the left to the
% right of the split 
% in that case send the left and the right node out for inquiry 
if ( ((point(tree_cell(node_number).splitdim)+range(1,tree_cell(node_number).splitdim))<=tree_cell(node_number).splitval) &&...
        ((point(tree_cell(node_number).splitdim)+range(2,tree_cell(node_number).splitdim))>tree_cell(node_number).splitval) )

    if (~isempty(tree_cell(node_number).left))
        [index_vals1,dist_vals1,vector_vals1]=kd_rangequery(0,point,range,tree_cell(node_number).left);
    else
        index_vals1=[];dist_vals1=[];vector_vals1=[];
    end
    if (~isempty(tree_cell(node_number).right))
        [index_vals2,dist_vals2,vector_vals2]=kd_rangequery(0,point,range,tree_cell(node_number).right);
    else
        index_vals2=[];dist_vals2=[];vector_vals2=[];
    end
    index_vals=[index_vals;index_vals1;index_vals2];
    dist_vals=[dist_vals;dist_vals1;dist_vals2];
    vector_vals=[vector_vals;vector_vals1;vector_vals2];

end

% after everything is done clear out the global variables 
if(nargin==3)

clear global tree_cell;
clear global safety_check;

end 

