% prepare inputs for homogenous covariance functions and their derivatives
% e.g. fcov, gcov, gTcov etc.

% sqd distances are supplied
sqd_diff_style = (length(varargin)==1);

if sqd_diff_style
    sqd_diffs = varargin{1};
    [NRows,NCols,NDims] = size(sqd_diffs);
    NData = NRows*NCols;
    sqd_diffs = reshape(sqd_diffs, NData, NDims);
    
    matrify_style = true;
else
    
    

    matrify_style = ~strcmpi(varargin{end},'vector');
    
    % ischar is slower
    %matrify_style = ~ischar(varargin{end});
    if ~matrify_style
        % we are not computing all cross-terms in the covariance, only the
        % covariance between varargin{1}(i,:) and varargin{2}(i,:) for all
        % i.
        

        [NData,NDims] = size(varargin{1});

        insL = varargin{1};
        insR = varargin{2};
    else
        % called in matrify style, with the first NDims matrices representing
        % the input dimensions 1:NDims for the row input vectors and the
        % remaining matrices the input dimensions for the column input vectors.

        %ins={[1111,1112;1121,1122],[1211,1212;1221,1222],[1311,1312;1321,1322],...
        %    [9111,9112;9121,9122],[9211,9212;9221,9222],[9311,9312;9321,9322]}
        %    ;

        NDims=length(varargin)/2;

        NRows=size(varargin{1},1);
        NCols=size(varargin{1},2);
        NData=NRows*NCols;




        ins=reshape(varargin,NDims,2); % break into L and R columns
        insL=[ins{:,1}];
        insL=reshape(insL,NData,NDims); % put rows and cols into a single vector
        insR=[ins{:,2}];
        insR=reshape(insR,NData,NDims);

        % left are vectors drawn solely from rows, and right solely from
        % columns, but length(insL) != NRows, obviously
    end
    sqd_diffs = (insL-insR).^2;
end

InputScale=ones(1,NDims);
OutputScale=1;
param=[];
per_roughness = 1;
if ~isempty(hp)
    InputScale=hp{1};
    if length(InputScale) ~= NDims && ...
            (~iscell(fnName) || ...
            (iscell(fnName) && ...
            ~(strcmpi(fnName{2},'great-circle') || strcmpi(fnName{2},'sets'))))
        error('Dimensions of input scale and inputs do not match.')
    end
    if length(hp)>1
        OutputScale=hp{2};
        if length(OutputScale)>1
            OutputScale = reshape(OutputScale,NData,1);
        end
        if length(hp)>2
            per_roughness=hp{3};
        end
    end
end

if iscell(fnName) && ~all(cellfun(@ischar,fnName))
    param_ind = cellfun(@(x) ~ischar(x),fnName);
    param = fnName{param_ind};
    fnName = fnName(~param_ind);
    if length(fnName) == 1
        fnName = fnName{1};
    end
end

if iscell(fnName)
    switch fnName{2}
        case 'periodic'
           
            arg = pi*bsxfun(@rdivide,sqrt(sqd_diffs),InputScale);
            
            r = (1/per_roughness)*...
                sqrt(sin(arg).^2*ones(NDims,1));
            
        case 'great-circle'
            r=abs(distance(insL(:,1),insL(:,2),insR(:,1),insR(:,2))/InputScale);
        case 'sets'
            
            %The inputs are of the form 
            % [x^1_1, ..., x^1_{dim_elem}, w^1, 
            % x^1_2, ..., x^2_{dim_elem}, w^2, 
            %   ...
            % x^1_{card_set}, ..., x^{card_set}_{dim_elem}, w^{card_set}]
            
            % InputScale(1:dim_elem) give input scales for inputs
            % 1:dim_elem, InputScale(end) gives an overall scale for the
            % set distance.
            
            % the length of each vector representing an element of one of
            % our sets
            dim_elem = param;
            
            % assume sets of equal size, cardinality equal to
            card_set = NDims/(dim_elem + 1);
            
            if card_set ~= round(card_set)
                error('there are not a round number of elements in each set')
            end
            

            
            weightsL = insL(:,(dim_elem + 1):dim_elem:end);
            weightsR = insL(:,(dim_elem + 1):dim_elem:end);
            
            r = nan(NData,1);
            for i = 1:NData
                
                sqd_distmat = zeros(dim_elem);
                for j = 1:dim_elem
                    xL = insL(i,j:(dim_elem + 1):end);
                    xR = insR(i,j:(dim_elem + 1):end);

                    % sqdist(xL,xR) is faster than bsxfun(@(x,y)
                    % (x-y).^2,xL,xR')
                    sqd_distmat = sqd_distmat + sqdist(xL,xR)./(InputScale(j)^2);
                end

                r(i) = emd_mex(weightsL(i,:), weightsR(i,:), sqrt(sqd_distmat));
            end
            r = r./InputScale(end);
    end
    hom_fn_Name=fnName{1};
else % assume non-periodic
    hom_fn_Name = fnName;
    r=sqrt(sqd_diffs*InputScale'.^-2);
end
