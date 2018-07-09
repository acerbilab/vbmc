function [indices] = find_needle(haystack, needle)
indices = nan(size(needle,1),1);
cur_needle = 1;
cur_index = 1;
for cur_haystack = 1:size(haystack,1)
   if all(haystack(cur_haystack,:) == needle(cur_needle,:))
       indices(cur_index) = cur_haystack;
       cur_needle = cur_needle + 1;
       cur_index = cur_index + 1;
   end
   if cur_needle>size(needle,1)
       break
   end
end
if cur_needle < size(needle,1)
   fprintf('Failed to find all needles in the haystack: The last %d are missing.\n', size(needle,1)-cur_needle);
end