% function [farthest, enough] = find_farthest(points, box, requested)
%
% finds points in a box farthest away from a given set of points,
% in the sense of having maximum minimum distance.
%
% _arguments_
%    points: an nxd set of d-dimensional points
%       box: a 2xd set of bounds for the box of interest
% requested: the number of farthest points desired
%
% _returns_
% farthest: the farthest points
%   enough: a boolean flag indicating whether the full number of
%           requested points were found
%
% author: roman garnett
%   date: 28 june 2008

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

function [farthest, pc_distances, enough] = find_farthest(points, box, requested, length_scales, ss_distance)%, ascend)

num_dims=size(points,2);



% % Try without voronoi
% inds=find(triu(true(num_points)));
% xinds=rem(inds,num_points);
% xinds(xinds==0)=num_points;
% yinds=(inds-xinds)/num_points+1;
% 
% candidates=0.5*(points(xinds,:)+points(yinds,:));
% candidates=[candidates;corners];


if num_dims>1
    
    try
        if nargin<4
            length_scales=ones(1,size(points,2));
        end

        num_points=size(points,1);
        num_box=size(box,1);

        scaled_points=points./repmat(length_scales,num_points,1);
        scaled_box=box./repmat(length_scales,num_box,1);

        corners = allcombs(scaled_box);
        corners_and_points=unique(roundto([corners; scaled_points],2), 'rows');
    
        vertices = voronoin(corners_and_points,{'QJ'}); % This QJ seems to remove errors like voronoi(allcombs(rand(2,n)))

        num_vertices = size(vertices, 1);

        % some points returned by voronoin may be outside the given box
        legal = prod((vertices <= repmat(max(scaled_box), num_vertices, 1)) .* ...
                     (vertices >= repmat(min(scaled_box), num_vertices, 1)), 2);

        scaled_candidates = unique([corners; vertices(legal > 0, :)], 'rows');
        Ncandidates = size(scaled_candidates,1);
        candidates = scaled_candidates.*repmat(length_scales,Ncandidates,1);

        % may want pc_distances even if there aren't enough

        pc_distances = sqrt(squared_distance(points, candidates, length_scales));

%         for i=1:num_dims;
%             i2s = [1:i-1,i+1:num_dims];
%             i2 = i2s(ceil(rand*(num_dims-1)));
%             figure;plot(points(:,i),points(:,i2),'k.','MarkerSize',14)
%             hold on;plot(candidates(:,i),candidates(:,i2),'m.','MarkerSize',10)
%         end
        
        num_missing = requested - size(candidates, 1);
        enough = num_missing <= 0;
        if (~enough)
            
            % generate points on a unit circle
            directions = rand(num_missing, num_dims)-0.5;
            directions_length = sqrt(sum(directions.^2, 2));
            directions = bsxfun(@rdivide, directions, directions_length);
            % scale by length_scales
            directions = bsxfun(@times, directions, length_scales);
            
            start_inds = ceil(num_points * rand(num_missing, 1));
            starts = points(start_inds,:);
            
            extra_candidates = starts + directions;
            
            farthest = [candidates;extra_candidates];
            
            return;
        end

        % if ascend
        %     [y, indices] = sort(pc_distances, 'ascend');
        %     farthest = candidates(indices(end-requested+1:end), :);
        % else

            [y, indices] = sort(sum(pc_distances), 'descend');
        if ~isempty(requested)
            farthest = candidates(indices(1:requested), :);
            pc_distances = pc_distances(:,indices(1:requested));
        else
            farthest = candidates(indices, :);
            pc_distances = pc_distances(:,indices);
        end
    catch e
        farthest = far_pts(points, box, requested);
    end
    
else
    %vertices=0.5*(corners_and_points(2:end)+corners_and_points(1:end-1)); 
    %vertices=(min(corners):ss_distance:max(corners))';
    
    if nargin<5
        ss_distance = sqrt(squared_distance(points,points,1));
    end
    
    ss_distance=ss_distance*length_scales;
    
%     low=min(box)-ss_distance;
%     high=max(box)+ss_distance;
%     
%     corners_and_points=unique(roundto([low; points; high],2), 'rows');
%     
%     gaps=diff(corners_and_points);
%     verts_in_gap=zeros(length(gaps),1);
%     [max_gap,max_ind]=max(gaps);
%     while max_gap>2*ss_distance && sum(verts_in_gap)<requested
%         verts_in_gap(max_ind)=verts_in_gap(max_ind)+1;
%         gaps(max_ind)=gaps(max_ind)-ss_distance;
%         [max_gap,max_ind]=max(gaps);
%     end
%     farthest=[];
%     for i=1:length(gaps)
%         verts=linspace(corners_and_points(i),corners_and_points(i+1),verts_in_gap(i)+2)';
%         farthest=[farthest;verts(2:end-1);];
%     end
    points = unique([points(:); box(:)]);
    gaps = diff(points);    

    num_in_gaps = zeros(size(gaps));

    num_added = 0;
    while (num_added < requested)
     [best index] = max(gaps ./ (num_in_gaps + 2));

     if (best < ss_distance)
       break;
     end

     num_in_gaps(index) = num_in_gaps(index) + 1;
     num_added = num_added + 1;
    end

    farthest = zeros(num_added, 1);
    filled = find(num_in_gaps);

    last = 1;
    for i = 1:length(filled)
     filler_points = ...
       linspacey(points(filled(i)), points(filled(i) + 1),num_in_gaps(filled(i)) + 2);
     farthest(last:(last + num_in_gaps(filled(i)) - 1)) = filler_points(2:end - 1);
     last = last + num_in_gaps(filled(i));
    end
    
    pc_distances = sqrt(squared_distance(points, farthest, length_scales));
end

    
%end
