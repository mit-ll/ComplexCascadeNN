%===========================================================
function [w,b]=mycalcnw(pr,s,n)
%CALCNW Calculates Nugyen-Widrow initial conditions.
%
%  PR
%  S - Number of neurons.
%  N - Active region of transfer function N = [Nmin Nmax].

% Force spread across hardlim and hardlims
if (n(2) == n(1))
  n = n + [-1 1];
end

% Special case: No inputs
r = size(pr,1);
if (r == 0) || (s == 0)
  w = zeros(s,r);
  b = zeros(s,1);
  return
end

% Fix nonfinite pr
i = find(~isfinite(pr(:,1)+pr(:,2)));
pr(i,:) = repmat([0 1],length(i),1);

% Remove constant inputs
R = r;
ind = find(pr(:,1) ~= pr(:,2));
r = length(ind);
pr = pr(ind,:);

% Special case: No variable inputs
if (r == 0)
  w = zeros(s,R);
  b = zeros(s,1);
  return
end

% Nguyen-Widrow Method
% Assume inputs and net inputs range in [-1 1].
% --------------------

wMag = 0.7*s^(1/r);
wDir = randnr(s,r);
w = wMag*wDir;

if (s==1)
  b = 0;
else
  b = wMag*linspace(-1,1,s)'.*sign(w(:,1));
end

% --------------------

% Conversion of net inputs of [-1 1] to [Nmin Nmax]
x = 0.5*(n(2)-n(1));
y = 0.5*(n(2)+n(1));
w = x*w;
b = x*b+y;

% Conversion of inputs of PR to [-1 1]
x = 2./(pr(:,2)-pr(:,1));
y = 1-pr(:,2).*x;
xp = x';
b = w*y+b;
w = w.*xp(ones(1,s),:);

% Replace constant inputs
ww = w;
w = zeros(s,R);
w(:,ind) = ww;

end
%===========================================================