% function covvyout = manage_hyper2_samples(covvy, method)
%
% handles the management of the locations of hyper2samples
% according to various methods
%
% _arguments_
%
%   covvy: the covariance structure of interest
%  method: a string representing the method to use:
%          'rotating': moves through the hyper2samples one each
%                      time step and performs gradient ascent
%               other: does not move the hyper2samples
%
% _returns_
%
%  covvyout: a covariance structure with the (possibly) modified
%            hyper2samples
%
% author: roman garnett
%   date: 20 august 2008

% Copyright (c) 2008, Roman Garnett <rgarnett@robots.ox.ac.uk>
% 
% Permission to use, copy, modify, and/or distribute this software for any
% purpose with or without fee is hereby granted, provided that the above
% copyright notice and this permission notice appear in all copies.
% 
% THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
% WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
% MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
% ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
% WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
% ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
% OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.

function covvyout = manage_hyper2_samples(covvy, stage)

% the step size to use for any gradient ascent action, who knows what
% this should be
if ~isfield(covvy,'gradient_ascent_step_size')
    covvy.gradient_ascent_step_size = 0.1;
end

if ~isfield(covvy,'debug')
    covvy.debug = false;
end
debug = covvy.debug;

step_size = covvy.gradient_ascent_step_size;

covvyout = covvy;

% any lower than this will give us covariance functions that are
% numerically infinite at their peaks, leading to all kinds of mischief
min_logscale=covvy.min_logscale;

max_logscale=covvy.max_logscale;

scales=[covvy.samplesSD{:}];

switch lower(stage)
    case 'move'
        % shift the to be moved hyper2 samples by one small step of gradient
        % ascent
        
        num_hps=numel(covvy.hyperparams);
        num_h2s=numel(covvy.hyper2samples);
        active_hp_inds=covvy.active_hp_inds;
        
        toMove = get_hyper2_samples_to_move(covvy);
        if isfield(covvy,'ignoreHyper2Samples')
            toMove = setdiff(toMove,covvy.ignoreHyper2Samples);
        end
        
        locations=cat(1,covvyout.hyper2samples(toMove).hyper2parameters);
        gradients_full=zeros(length(toMove),num_hps);
        tilda_gradients_full=zeros(length(toMove),num_hps);
        
        if  strcmp(covvy.manage_h2s_method,'optimal')
            tilda_gradients=cell2mat(cat(2, ...
                covvy.hyper2samples(toMove).tilda_glogL))';
            gradients=cell2mat(cat(2, covvy.hyper2samples(toMove).glogL))';
            
            tilda_logLs = cat(1,covvy.hyper2samples(toMove).tilda_logL);
            logLs = cat(1,covvy.hyper2samples(toMove).logL);
            
            tilda_gradients_full(:,active_hp_inds)=tilda_gradients;
            gradients_full(:,active_hp_inds)=gradients;
            
            [tilda_ascended,tilda_ascended_tilda_logLs] = ...
                gradient_ascent(locations, tilda_logLs, ...
                tilda_gradients_full, step_size, scales);            
            [ascended,ascended_logLs] = gradient_ascent(locations, ...
                logLs, gradients_full, step_size, scales);
            
            tilda_ascended_logLs = logLs + sum(gradients_full .* ...
                (tilda_ascended - locations),2);
            ascended_tilda_logLs = tilda_logLs + ...
                sum(tilda_gradients_full .* (ascended - locations),2);
            
            % We determine a score, which is the sum of the expected maxima
            % of logL and tilda_logL after the move
            
            [max_tilda_ascended_tilda_logLs, max_tilda_ind] = ...
                max(tilda_ascended_tilda_logLs);
            all_but_max_tilda_ind = setdiff(1:length(toMove),max_tilda_ind);
             if ~isempty(all_but_max_tilda_ind)
                tilda_score = max_tilda_ascended_tilda_logLs + ...
                    max(tilda_ascended_logLs(max_tilda_ind),...
                    max(logLs(all_but_max_tilda_ind)));
            else
                tilda_score = max_tilda_ascended_tilda_logLs + ...
                    tilda_ascended_logLs(max_tilda_ind);      
            end
            
            
            [max_ascended_logLs, max_ind] = max(ascended_logLs);
            all_but_max_ind = setdiff(1:length(toMove),max_ind);
            if ~isempty(all_but_max_ind)
                score = max_ascended_logLs + ...
                    max(ascended_tilda_logLs(max_ind),...
                    max(tilda_logLs(all_but_max_ind)));
            else
                score = max_ascended_logLs + ascended_tilda_logLs(max_ind);      
            end

            scale_tilda_likelihood = tilda_score>=score;
            if scale_tilda_likelihood
                if (debug); disp('scale tilda likelihood'); end
                toMove = toMove(max_tilda_ind);
                ascended = tilda_ascended(max_tilda_ind,:);
            else
                if (debug); disp('scale likelihood'); end
                toMove = toMove(max_ind);
                ascended = ascended(max_ind,:);
            end  
            
        else
            
            if isfield(covvy,'scale_tilda_likelihood')
                scale_tilda_likelihood = covvy.scale_tilda_likelihood;
            else
                scale_tilda_likelihood = true;
            end
            
            if scale_tilda_likelihood
                gradients=cell2mat(cat(2, ...
                    covvyout.hyper2samples(toMove).tilda_glogL))';
            else
                gradients=cell2mat(cat(2, ...
                    covvyout.hyper2samples(toMove).glogL))';
            end

            gradients_full(:,active_hp_inds)=gradients;

            gradient_ascent(locations, [], gradients_full, step_size, ...
                scales);
            
        end
        





        
if (debug); toMove
end
        ascended = min(max(ascended,min_logscale),max_logscale);
        ind = 0;
        for i = toMove	
            ind = ind+1;
            covvyout.hyper2samples(i).hyper2parameters = ascended(ind,:);
        end

        covvyout.lastHyper2SamplesMoved = toMove;    
    case 'clear'
        % remove now incorrect fields
        
        Moved = covvy.lastHyper2SamplesMoved;
        
        for i = Moved	
            fields = fieldnames(covvyout.hyper2samples(i));
            for j = 1:length(fields)
                covvyout.hyper2samples(i).(fields{j})=[];
            end
            covvyout.hyper2samples(i).hyper2parameters=...
                covvy.hyper2samples(i).hyper2parameters;
        end
end

