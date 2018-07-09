function M=tensify(f,rows,cols,stacks,Params)
% Matrify returns the outer product defined by f with parameters Params.
% rows and cols are both column matrices, respectively the vectors
% containing the first and second inputs to the outer product. Hence,
% output M is the matrix of f evaluated at each possible pair of first and
% second inputs. rows and cols must be *columns* of arguments to f. If th
% inputs to f are multiple dimensional, they must instead be treated as
% multiple inputs - f(r1,r2,c1,c2); If the outputs of f are
% multidimensional as [y1,y2,y3,...]=f(rows,cols) then M will be
% [y1,y2,y3,...]. That is, there will be a matrix over rows and cols for
% the first output dimension, following by a matrix over rows and cols for
% the second output dimension, and so on.

% tic;matrify(@(x,y) [normpdf(x,y,1)],(1:1000)',(1:1000)');toc
% <
% tic;bsxfun(@(x,y) [normpdf(x,y,1)],(1:1000)',(1:1000));toc
% Hooray!

% if length(f)==1
%     Sparsity=false;
% elseif length(f)==2
%     % f is a cell whose first element is the relevant function
%     f=f{1};
%     % and whose second element is the threshold below which the output of f
%     % will be treated as exactly zero, allowing a sparse representation
%     Threshold=f{2};
%     Sparsity=true;
% end

if size(rows,1)==0 || size(cols,1)==0 || size(stacks,1)==0
    M=rows*cols'; % could cause problems if called with an x by zero input
    return
end
  
N=size(rows,2);
if nargin==3 % No Parameters supplied    
     
    Args=cell(1,3*N);
    for n=1:N
        [Args{n+2*N},Args{n+N},Args{n}]=meshgrid(stacks(:,n),cols(:,n),rows(:,n));
    end
    M=f(Args{:});
    
%     if size(rows,2)==1
%         [COLS,ROWS]=meshgrid(cols,rows);
%         M=f(ROWS,COLS);
%     elseif size(rows,2)==2
%         [COLS1,ROWS1]=meshgrid(cols(:,1),rows(:,1));
%         [COLS2,ROWS2]=meshgrid(cols(:,2),rows(:,2));
%         
%         M=f(ROWS1,ROWS2,COLS1,COLS2);
%     end
    
elseif nargin==4 % Parameters for f supplied
    
    Args=cell(1,3*N+1);
    for n=1:N
        [Args{n+2*N},Args{n+N},Args{n}]=meshgrid(stacks(:,n),cols(:,n),rows(:,n));
    end
    Args{3*N+1}=Params;
    M=f(Args{:});
    
%     if size(rows,2)==1
%         [COLS,ROWS]=meshgrid(cols,rows);
%         M=f(ROWS,COLS,Params);
%     elseif size(rows,2)==2
%         [COLS1,ROWS1]=meshgrid(cols(:,1),rows(:,1));
%         [COLS2,ROWS2]=meshgrid(cols(:,2),rows(:,2));
%         M=f(ROWS1,ROWS2,COLS1,COLS2,Params);
%     end
end

% if Sparsity
%     % Treat as exactly zero all elements less than Threshold
%     M=sparse(M.*(M>=Threshold));
% end


