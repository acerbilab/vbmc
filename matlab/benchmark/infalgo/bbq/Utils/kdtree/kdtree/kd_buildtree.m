function tree_output = kd_buildtree(X,plot_stuff,parent_number,split_dimension)

% pramod vemulapalli 02/08/2010
% inspired by the work done by Jan Nunnink, 2003.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% INPUTS
% X                --- contains the data (nxd) matrix , where n is the
%                      number of feature vectors and d is the dimensionality 
%                      of each feature vector 
% parent_number    --- Internal variable .... Donot Assign 
% split_dimension  --- Internal variable .... Donot Assign 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% OUTPUTS
% tree_output --- contains the a array of structs, each struct is a node 
%                 node in the tree. The size of the array is equal to the 
%                 the number of feature vectors in the data matrix X 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% each struct in the output cell array contains the following information 
% left        -  (tree) left tree node location in the array 
% right       -  (tree) right tree node location in the array 
% numpoints   -  number of points in this node 
% nodevector  -  the median of the data along dimension that it is split 
% hyperrect   -  (2xd) hyperrectangle bounding the points 
% type        -  'leaf' = node is leaf
%                'node' = node has 2 children
% parent      -  the location of the parent of the current node in the
%                struct array 
% index       -  the index of the feature vector in the original matrix 
%                that was used to build the tree
% splitdim    -  the dimension along which the split is made 
% splitval    -  the value along that dimension in which the split is made 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

global tree_cell;
global node_number;
global safety_check;

if nargin ==2
    
    safety_check=0;
    
    % add the index values to the last column
    % easy way to keep track of them 
    [n,d] = size(X);
    X(:,d+1)=1:n';
    node_number=1;
    split_dimension=0;
    parent_number=0;
    
    % intialize the node 
    Node.type='node'; 
    Node.left=0;
    Node.right=0;
    Node.nodevector=zeros(1,d);
    Node.hyperrect=[zeros(1,d);zeros(1,d)];
    Node.numpoints=0;
    Node.index=0;
    Node.parent=0;
    Node.splitval=0;
    
    % initilaze the tree 
    hold_cell_data(1:n)=Node;
    tree_cell=hold_cell_data;
    clear hold_cell_data;
    
else
    
    [n,d] = size(X(:,1:end-1));
    node_number=node_number+1;
    split_dimension=split_dimension+1;
    
end

if (isempty(safety_check))    
    error ('Some thing is wrong with the number of inout variables .... Please check');
end 

if n==0 
    fprintf('Error: 0 points in node, causing endless loop, press ctrl-C.\n'); 
end

assigned_nn=node_number; % assigned node number for this particular iteration 

% set assignments to current node 
tree_cell(assigned_nn).type='node'; 
tree_cell(assigned_nn).parent=parent_number; 


% if there is 1 datapoint left, make a leaf out of it 
if n==1
    tree_cell(assigned_nn).left=[];
    tree_cell(assigned_nn).right=[];
    tree_cell(assigned_nn).nodevector=X(:,1:end-1);
    tree_cell(assigned_nn).hyperrect=[X(:,1:end-1);X(:,1:end-1)];
    tree_cell(assigned_nn).type='leaf';
    tree_cell(assigned_nn).numpoints=1;
    tree_cell(assigned_nn).index=X(:,end);

    if plot_stuff==1	kd_plotbox(assigned_nn,'node'); end
    tree_output=assigned_nn;
    return;
end

% if there are more than 1 data points left calculate some of the node
% values 
tree_cell(assigned_nn).numpoints = n;
a = min(X(:,1:end-1)); b = max(X(:,1:end-1));
tree_cell(assigned_nn).hyperrect = [a; b];

% if the feature vectors happen to be the same then leaf again 
if a==b tree_cell(assigned_nn).type='leaf'; end

 
% recursively build rest of tree
% if the node is a leaf then assign the index and node vector 
if (strcmp(tree_cell(assigned_nn).type,'leaf'))

    tree_cell(assigned_nn).nodevector = mean(X(:,1:end-1));
    tree_cell(assigned_nn).index=X(:,end);
    if plot_stuff==1 kd_plotbox(assigned_nn,'node'); end
    
else

    % if it is not a leaf 
    
    % figure out which dimension to split (going in order)
    tree_cell(assigned_nn).splitdim=mod(split_dimension,d)+1;
    % figure out the median value to cut this dimension 
    median_value=median(X(:,tree_cell(assigned_nn).splitdim));
    % find out all the values that are lower than or equal to this median 
    i=find(X(:, tree_cell(assigned_nn).splitdim)<=median_value);

    
    % if there are more than 2 points 
    if (tree_cell(assigned_nn).numpoints>2)
        % as there more than two points 
        [max_val,max_pos]=max(X(i, tree_cell(assigned_nn).splitdim));
        % recurse for everything to the left of the median 
        tree_cell(assigned_nn).left = kd_buildtree(X([i(1:max_pos-1);i(max_pos+1:end)], :),plot_stuff, assigned_nn,split_dimension);
        % recurse for everything to the right of the median
        if(size(X,1)>size(i,1))
            tree_cell(assigned_nn).right = kd_buildtree(X(find(~(X(:, tree_cell(assigned_nn).splitdim)<=median_value)), :),plot_stuff,assigned_nn,split_dimension);
        else
            tree_cell(assigned_nn).right =[];
        end
    else
        % if there are only two data points left 
        % choose the left value as the median 
        % make the right value as a leaf 
        % leave the left leaf blank 
        [max_val,max_pos]=max(X(i, tree_cell(assigned_nn).splitdim));
        if (i(max_pos)==1); min_pos=2; else min_pos=1; end 
        tree_cell(assigned_nn).left = [];
        tree_cell(assigned_nn).right = kd_buildtree(X(min_pos,:),plot_stuff,assigned_nn,split_dimension);
    end

    % assign the median vector to this node 
    tree_cell(assigned_nn).nodevector = X(i(max_pos),1:end-1);
    tree_cell(assigned_nn).index=X(i(max_pos),end);
    tree_cell(assigned_nn).splitval=X(i(max_pos), tree_cell(assigned_nn).splitdim);

    % plot stuff if you want to 
    if plot_stuff==1 kd_plotbox(assigned_nn,'node'); end
    if plot_stuff==1 kd_plotbox(assigned_nn,'box'); end

end


% final clean up 
if nargin ==2
    % As all computation is complete then return tree structure 
    % and clear other data 
    tree_output=tree_cell;
    clear global tree_cell;
else
    % otherwise output the assigned_nn for storage in the parent 
    tree_output=assigned_nn;
end
