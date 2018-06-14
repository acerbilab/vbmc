function K = derivativise(covfn,flag)
% a wrapper to convert any cov fn into one appropriate for derivative
% observations.
% example usage:
% GP.covfn = @(flag) derivativise(@(flag) hom_cov_fn(hps_struct,type,flag),flag);

if nargin<2
    flag='plain';
end

K=@(hp,XsL,XsR) derivativised_cov(covfn,flag,XsL,XsR,hp);



function Kmat = derivativised_cov(covfn,flag,XsL,XsR,hp)
% the last variable is an integer giving the index of the variable that the
% derivative is taken with respect to. If 0, no derivative is taken.

indicesL = XsL(:,end);
indicesR = XsR(:,end);

XsL = XsL(:,1:(end-1));
XsR = XsR(:,1:(end-1));

obsL=indicesL==0;
obsR=indicesR==0;

some_obsL=~all(~obsL);
some_obsR=~all(~obsR);

derivsL=setdiff(unique(indicesL),0);
derivsR=setdiff(unique(indicesR),0);

gobsL=false(size(XsL,1),max(derivsL));
for derivL_ind = 1:length(derivsL)
    derivL=derivsL(derivL_ind);
    gobsL(:,derivL)= indicesL==derivL;
end
gobsR=false(size(XsR,1),max(derivsR));
for derivR_ind = 1:length(derivsR)
    derivR=derivsR(derivR_ind);
    gobsR(:,derivR)= indicesR==derivR;
end

[flag, grad_hp_inds] = process_flag(flag);
switch flag
    case 'vector'
        K = covfn('plain');
        DK = covfn('grad inputs');
        DDK = covfn('hessian inputs');
        
        Kmat = nan(size(XsL,1),size(XsR,1));
        if some_obsL && some_obsR
            Kmat(obsL,obsR)=K(hp,XsL(obsL,:),XsR(obsR,:));
        end
        if some_obsL 
            for derivR_ind = 1:length(derivsR); 
                derivR=derivsR(derivR_ind);
                derivs_cell = DK(hp, XsL(obsL,:),XsR(gobsR(:,derivR),:));
                Kmat(obsL,gobsR(:,derivR)) = -derivs_cell{derivR};
            end
        end
        if some_obsR
            
            for derivL_ind = 1:length(derivsL)
                derivL=derivsL(derivL_ind);
                derivs_cell = DK(hp,XsL(gobsL(:,derivL),:),XsR(obsR,:));
                Kmat(gobsL(:,derivL),obsR)=derivs_cell{derivL};
            end
        end
        
            for derivL_ind = 1:length(derivsL); 
                derivL=derivsL(derivL_ind);
                for derivR_ind = 1:length(derivsR)
                    derivs_cell = DDK(hp,...
                        XsL(gobsL(:,derivL),:),XsR(gobsR(:,derivR),:));

                    
                    derivR=derivsR(derivR_ind);
                    Kmat(gobsL(:,derivL),gobsR(:,derivR)) = ...
                        -derivs_cell{derivL,derivR};
                end
            end
            
        % clearly this is vastly inefficient
        Kmat = diag(Kmat);
        
    case 'plain'
        K = covfn('plain');
        DK = covfn('grad inputs');
        DDK = covfn('hessian inputs');
        
        Kmat = nan(size(XsL,1),size(XsR,1));
        if some_obsL && some_obsR
            Kmat(obsL,obsR)=K(hp,XsL(obsL,:),XsR(obsR,:));
        end
        if some_obsL 
            for derivR_ind = 1:length(derivsR); 
                derivR=derivsR(derivR_ind);
                derivs_cell = DK(hp, XsL(obsL,:),XsR(gobsR(:,derivR),:));
                Kmat(obsL,gobsR(:,derivR)) = -derivs_cell{derivR};
            end
        end
        if some_obsR
            
            for derivL_ind = 1:length(derivsL)
                derivL=derivsL(derivL_ind);
                derivs_cell = DK(hp,XsL(gobsL(:,derivL),:),XsR(obsR,:));
                Kmat(gobsL(:,derivL),obsR)=derivs_cell{derivL};
            end
        end
        
            for derivL_ind = 1:length(derivsL); 
                derivL=derivsL(derivL_ind);
                for derivR_ind = 1:length(derivsR)
                    derivs_cell = DDK(hp,...
                        XsL(gobsL(:,derivL),:),XsR(gobsR(:,derivR),:));

                    
                    derivR=derivsR(derivR_ind);
                    Kmat(gobsL(:,derivL),gobsR(:,derivR)) = ...
                        -derivs_cell{derivL,derivR};
                end
            end
            
    case 'grad inputs'
        K = covfn('grad inputs');
        DK = covfn('hessian inputs');
        
        L1 = size(XsL,1);
        L2 = size(XsR,1);
        num_dims = size(XsL,2);

        Kmat = mat2cell2d(zeros(num_dims*L1,L2),L1*ones(num_dims,1),L2);
        if some_obsL && some_obsR
            Kmat = cellfun(@(big_mat,little_mat) ...
                put_in(big_mat,little_mat,obsL,obsR), ...
                Kmat,K(hp,XsL(obsL,:),XsR(obsR,:)),'UniformOutput',false);
        end
        if some_obsL 
            for derivR_ind = 1:length(derivsR); 
                derivR=derivsR(derivR_ind);
                
                DKmat = DK(hp,XsL(obsL,:),XsR(gobsR(:,derivR),:));
                DKmat = DKmat(:,derivR);
                
                Kmat = cellfun(@(big_mat,little_mat) ...
                    put_in(big_mat,-little_mat,obsL,gobsR(:,derivR)), ...
                    Kmat,DKmat,'UniformOutput',false);
            end
        end
        if some_obsR
            for derivL_ind = 1:length(derivsL); 
                derivL=derivsL(derivL_ind);
                
                DKmat = DK(hp,XsL(gobsL(:,derivL),:),XsR(obsR,:));
                DKmat = DKmat(derivL,:);
                
                Kmat = cellfun(@(big_mat,little_mat) ...
                    put_in(big_mat,little_mat,gobsL(:,derivL),obsR), ...
                    Kmat,DKmat,'UniformOutput',false);
            end
        end
            
    case 'grad hyperparams'
        
        K = covfn('grad hyperparams');
        DK = covfn('grad hyperparams grad inputs');
        DDK = covfn('grad hyperparams hessian inputs');
        
        L1 = size(XsL,1);
        L2 = size(XsR,1);
        num_hps = length(hp);

        Kmat = mat2cell2d(zeros(num_hps*L1,L2),L1*ones(num_hps,1),L2);
        if some_obsL && some_obsR
            Kmat = cellfun(@(big_mat,little_mat) ...
                put_in(big_mat,little_mat,obsL,obsR), ...
                Kmat,K(hp,XsL(obsL,:),XsR(obsR,:)),'UniformOutput',false);
        end
        if some_obsL 
            for derivR_ind = 1:length(derivsR); 
                derivR=derivsR(derivR_ind);
                Kmat = cellfun(@(big_mat,little_mat) ...
                    put_in(big_mat,-little_mat,obsL,gobsR(:,derivR)), ...
                    Kmat,DK(hp,XsL(obsL,:),XsR(gobsR(:,derivR),:)),'UniformOutput',false);
            end
        end
        if some_obsR
            for derivL_ind = 1:length(derivsL); 
                derivL=derivsL(derivL_ind);
                Kmat = cellfun(@(big_mat,little_mat) ...
                    put_in(big_mat,little_mat,gobsL(:,derivL),obsR), ...
                    Kmat,DK(hp,XsL(gobsL(:,derivL),:),XsR(obsR,:)),'UniformOutput',false);
            end
        end

            for derivL_ind = 1:length(derivsL); 
                derivL=derivsL(derivL_ind);
                for derivR_ind = 1:length(derivsR); 
                    derivR=derivsR(derivR_ind);
                    Kmat = cellfun(@(big_mat,little_mat) ...
                        put_in(big_mat,-little_mat,gobsL(:,derivL),gobsR(:,derivR)), ...
                        Kmat,DDK(hp,XsL(gobsL(:,derivL),:),XsR(gobsR(:,derivR),:)),'UniformOutput',false);
                end
            end
end


function mat1 = put_in(mat1,mat2,xinds,yinds)
mat1(xinds,yinds) = mat2;
