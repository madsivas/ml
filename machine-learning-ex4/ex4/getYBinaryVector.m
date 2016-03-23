% y = [ 2; 3; 1; 8; 5; 9; 0; 2 ] % size 8 vec
function [ybin] = getYBinaryVector(y, num_labels)
   m = size(y, 1);
   K = num_labels;
   ybin = zeros(m, K);
   for yi = 1 : size(y, 1)
      if y(yi, 1) == 0
         % yi
         % y(yi, 1)
         ybin(yi, K) = 1;
         continue;
      endif

      for idx = 1 : K - 1 
         if y(yi, 1) == idx 
            % y(yi, 1)
            ybin(yi, idx) = 1;
         endif
      endfor % idx
   endfor % yi
%   ybin = ybin';

end
