function hleg = mf_legend(handles,latex_labels,Locn,width)
%prepare a legend for use in matlabfrags. handles can be {} if desired. Use
%set to adjust properties other than Locn as required.

if nargin<4
    width = 4;
end

if width <= 1
    width = 2;
end

N = length(latex_labels);

nums_mat = repmat((1:N)',1,width);
nums_mat = sum(nums_mat.*repmat(10.^[(width-1):-1:0],N,1),2);
c = num2cell(nums_mat,2)';
labels = cellfun(@(x) num2str(x),c,'UniformOutput',false);

line_returns = cellfun(@(x) strfind(x,'\\'),latex_labels,'UniformOutput',false);

for i=1:N
    if ~isempty(line_returns{i})
        R = length(line_returns{i})+1;
        big_mat = repmat(str2num(labels{i}),R,1);
        big_mat = big_mat - i + (1:R)';
        labels{i} = num2str(big_mat);
        latex_labels{i} = ['\begin{tabular}{@{}l@{}}',latex_labels{i},'\end{tabular}'];
    end
end

hands_labels = [handles,labels];

hleg = legend(hands_labels{:},'Location',Locn);%,'Orientation','Horizontal')
%set(hleg,'XColor',[1 1 1],'YColor',[1 1 1])
legend boxoff
hlegc = get(hleg,'children');


for i = 1:N
        set( findobj(hlegc,'string',labels{i}), 'userdata',...
            ['matlabfrag:',latex_labels{i}]);
end
