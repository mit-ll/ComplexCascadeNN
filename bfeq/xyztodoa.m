function [vsAz, vsEl] = xyztodoa(uavbasisx,uavbasisy,uavbasisz,vstosource)
% Determine El and Az angles from UAV by considering
% vstosource=[cos(vsAz)*cos(vsEl) sin(vsAz)*cos(vsEl) sin(vsEl)]
vsEl = asin(vstosource'*uavbasisz);        %vsEl=90 is broadside
xpart = vstosource'*uavbasisx./cos(vsEl);
ypart = vstosource'*uavbasisy./cos(vsEl);
vsAzx = acos(xpart); %vsAzy = asin(ypart);
negypart = ypart<0;
vsAz = (2*pi - vsAzx).*negypart + vsAzx.*(1-negypart);

return;
