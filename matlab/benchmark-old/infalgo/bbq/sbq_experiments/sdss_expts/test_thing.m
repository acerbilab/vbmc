%matlabpool open

try
for i=1:5
    chol(-i); 
end
catch err
    getReport(err)
end