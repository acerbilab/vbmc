function [K,out2,out3] = simple2term_cov_fn(types,hp,flag)

type1=types{1};
type2=types{2};

T1=exp(hp(3));
L1=exp(hp(4));
T2=exp(hp(5));
L2=exp(hp(6));

if nargin<3
    flag='';
end

K=@(as,bs) matrify(@(at,bt)...
    fcov(type1,{T1,L1},at,bt)+fcov(type2,{T2,L2},at,bt),...
    as,bs);

switch flag
    case 'deriv inputs'
    DK1=@(as,bs) matrify(@(at,bt)...
            gcov(type1,{T1,L1},at,bt),as,bs);
    DK2=@(as,bs) matrify(@(at,bt)...
            gcov(type2,{T2,L2},at,bt),as,bs);
        
    out2=@(as,bs) cellfun(@plus,DK1(as,bs),DK2(as,bs));
        
    DDK1=@(as,bs) matrify(@(at,bt)...
            Hcov(type1,{T1,L1},at,bt),as,bs);
    DDK2=@(as,bs) matrify(@(at,bt)...
            Hcov(type2,{T2,L2},at,bt),as,bs);
 
    out3=@(as,bs) cellfun(@plus,DDK1(as,bs),DDK2(as,bs));
    
    case 'deriv hyperparams'
    DK=@(as,bs) [{zeros(size(as,1),size(bs,1));...          % mean
                    zeros(size(as,1),size(bs,1))};...         % logNoiseSD 
                    matrify(@(at,bt)...
                        gTcov(type1,{T1,L1},at,bt),as,bs);...
                    matrify(@(at,bt)...
                        2*fcov(type1,{T1,L1},at,bt),as,bs);...
                    matrify(@(at,bt)...
                        gTcov(type2,{T2,L2},at,bt),as,bs);...
                    matrify(@(at,bt)...
                        2*fcov(type1,{T2,L2},at,bt),as,bs)];
    out2=DK;
    case 'vector K'
        out2 = @(as,bs) fcov(type1,{T1,L1},as,bs,'vector style')+...
                fcov(type2,{T2,L2},as,bs,'vector style');
                    
end
