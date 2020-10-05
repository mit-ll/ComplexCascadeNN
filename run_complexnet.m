% some examples of use of complexnet class

addpath ./gradient-descent/



%% check for single complex layer

params.hiddenSize=[]; params.outputFcn='purelin';
params.trainFcn='trainlm';%'Adam2','Vanilla';
params.initFcn='c-nguyen-widrow';
params.domap = 0;
cnet = complexnet(params);
x=randn(1,100)+1i*randn(1,100);
%cnet.trainsingle(x,x*(3+5*1i) + (4-2*1i));
cnet = cnet.train(x,x*(3+5*1i) + (4-2*1i));
print(cnet)

%% check for single hidden real layer

params=[];
params.hiddenSize=[1];
params.layersFcn='purelin';
params.outputFcn='purelin';
params.trainFcn='RAdam';
params.initFcn='nguyen-widrow';
cnet = complexnet(params);
x=randn(1,100);
%cnet.trainsingle(x,3*x+2);
cnet = cnet.train(x,3*x + 2);
print(cnet);
aaa=cnet.test(x);
figure(1);plot(aaa,3*x+2);
xlabel('input'); ylabel('output');
%net = feedforwardnet(params.hiddenSize);
%net = train(net,x,3*x + 2)


%% check for pure phase output function

params.hiddenSize=[]; 
params.outputFcn='purephase'; 
net = complexnet(params);
x=randn(1,100)+1i*randn(1,100);
net.trainsingle(x,x./abs(x));
net.train(x,(3+3*1i)* x./abs(x));  % does not work - can't handle scalar
print(net);

%% check for split real/imaginary tansig(i.e. sigmoid) activation

params.hiddenSize=[]; params.outputFcn='sigrealimag'; net = complexnet(params)
x=randn(1,10000)+1i*randn(1,10000);
y = net.sigrealimag(x);
[W] = net.train(x,y);
out = net.test(x);
plot(real(y),real(out),'.'); hold on;
plot(imag(y),imag(out),'o');

% input 2 x output 10 checking dimensions of w and Levenberg-Marquardt
params.hiddenSize=[]; params.outputFcn='purelin';
params.batchtype = 'randperm';
params.minbatchsize=10;
params.trainFcn='trainlm';
net = complexnet(params)
x = randn(2,100)+1i*randn(2,100);
w = (randn(10,2)+1i*randn(10,2));
b = (randn(10,1)+1i*randn(10,1));
net.train(x, bsxfun(@plus,w * x,b));
net.Weights{1}
w
net.Weights{2}
b

%% check for multiple layers

% multiple real layers
params.hiddenSize=[1 2 2]; 
params.outputFcn='purelin'; 
params.trainFcn = 'Adadelta';%'trainlm';
params.layersFcn = 'sigrealimag2';
params.nbrofEpochs=1e4;
cnet = complexnet(params);
in = randn(1,200); 
out = [in; 3*in.^2 + 2*ones(size(in)); 10*in];
cnet = cnet.train(in,out);
outhat = cnet.test(in);
figure(101);clf;
plot(outhat.','.-'); hold on;   % can't get in^2 easily - use trainlm
plot(out.');

% check for multiple complex layers
params.hiddenSize=[1 2 2]; 
params.outputFcn='purelin';
params.trainFcn = 'Adadelta';%'trainlm';
params.layersFcn = 'sigrealimag2';
params.nbrofEpochs=1e4;
cnet = complexnet(params);
in=randn(1,200)+1i*randn(1,200); 
cnet = cnet.train(in,in*(3+5*1i) + (4-2*1i));
print(cnet)

% purephase multiple layers
params.hiddenSize=[1]; 
params.layersFcn='purephase'; 
params.outputFcn='purelin'; 
params.hiddenSize=[1 2 2]; 
params.trainFcn = 'Adadelta';%'trainlm';
params.nbrofEpochs=1e4;
cnet = complexnet(params);
in=randn(1,200)+1i*randn(1,200); 
cnet = cnet.train(in,(3+3*1i)* (in./abs(in)) );
print(cnet)

%% rotation example (as in Nitta)

% ellipse
% I
rot = pi/3; R = [cos(rot) -sin(rot); sin(rot) cos(rot)];
%R = diag([0.5 0.3])*R;

ini1 = linspace(-0.9,0.9,7); inr1 = zeros(size(ini1));
inr2 = linspace(-0.2,0.2,2); ini2 = 0.95*ones(size(inr2));
inr3 = inr2; ini3=-ini2;
shapeI=[inr1 inr2 inr3] + 1i*[ini1 ini2 ini3];

A = 0.6; B = 0.3; th = linspace(0,2*pi,220);
shapeO = A*cos(th) + 1i*B*sin(th);

shape = shapeI;
y= R * [real(shape); imag(shape)];
shaperotated = y(1,:) + 1i*y(2,:);

y= R * [real(shapeO); imag(shapeO)];
shapeOrotated = y(1,:) + 1i*y(2,:);

params.lrate=1e-2;
params.nbrofEpochs=1e4;
params.hiddenSize=[1 2 1];
params.domap=0;
params.layersFcn='sigrealimag2'; params.outputFcn='purelin';
% this example prefers crandn to nguyen-widrow, why?
params.initFcn='crandn';'c-nguyen-widrow';
params.minbatchsize=inf; params.batchtype = 'fixed';
params.trainFcn = 'trainlm';

cnet = complexnet(params);
cnet = cnet.train(shape,shaperotated);
outhat = cnet.test(shape);
outO = cnet.test(shapeO);
%print(cnet)

net = feedforwardnet( 2* params.hiddenSize);
realifyfn = @(x) [real(x); imag(x)];
net = train(net,realifyfn(shape),realifyfn(shaperotated));
outOr =net(realifyfn(shapeO)); outOr=outOr(1,:)+1i*outOr(2,:);

figure(123);clf;
plot(shape,'.','MarkerSize',12); hold on;
plot(shaperotated,'v','MarkerSize',12);
plot(outhat,'s','MarkerSize',12);

plot(shapeO,'.','MarkerSize',24);
plot(shapeOrotated,'o','MarkerSize',12);
plot(outO,'o','MarkerSize',12);
plot(outOr,'ro','MarkerSize',12);

legend('shape train','shape train rotated','net train output',...
    'shape test','shape test rotated', 'net test output','real net test output');
grid minor; grid;
boldify;


%% nonlinear volterra series
% n(t) = a(t) + sum beta * a(t-k) a(t-l)
at = randn(1,100000) + 0*1i*randn(1,100000);
L = length(at);
%beta = [ 0.3 + 0.7*1i, 0.7 - 0.3*1i]; lags = { [1,2] , [1,3] };
beta = [ 0.3 + 0*0.7*1i]; lags = { [1,2] };
numlags = max(max(lags{:}))+1; att = zeros(numlags,L);
for ll=0:numlags-1
    att(ll+1, ll + (1:L-ll) ) = at(1:L-ll);
end

nt = at;
txtvt='Volterra process a~N(0,1): n(0) = a(0)';
for term = 1:length(beta)
    nt = nt + beta(term) * ...
        att( 1+lags{term}(1),:).*att( 1+lags{term}(2),:);
    txtvt = sprintf('%s + (%0.3f + i%0.3f) a(-%d) a(-%d)',txtvt, ...
        real(beta(term)),imag(beta(term)),lags{term}(1),lags{term}(2));
end

% setup the samples available for prediction (up to numlags delays)
L = length(nt);
numlags = 6+1; ntt = zeros(numlags,L);
for ll=0:numlags-1
    ntt(ll+1, ll + (1:L-ll) ) = nt(1:L-ll);
end

traininds = 1:L/2; testinds=(L/2+1):L;
out = nt(traininds);
nttpast = ntt(2:end,traininds);
out1 = nt(testinds);
nttpast1 = ntt(2:end,testinds);

% linear predictor
%betahat = conj( (nttpast*nttpast') \ (nttpast * out') );
%outlinear = betahat'*nttpast;
params.hiddenSize = [];
params.layersFcn = 'purelin';params.outputFcn='purelin';
params.trainFcn = 'Adam2';
params.initFcn = 'nguyen-widrow';
params.minbatchsize = numel(traininds)/10;
net = complexnet(params);
net = net.train(nttpast,out);
outlinear1 = net.test(nttpast1);

% non-linear predictor
params=[];
params.lrate=1e-1;
params.domap = 1;
params.hiddenSize = [16 6 ];
params.debugPlots=0;
params.mu = 1e-3;
params.trainFcn = 'trainlm'; params.minbatchsize = round(numel(traininds)*0.7);
params.initFcn = 'nguyen-widrow';
params.batchtype='fixed';
params.layersFcn = 'mytansig'; %'mytanh';'sigrealimag2';
params.outputFcn = 'purelin';
params.nbrofEpochs = 300;
params.mu_inc = 10;
params.mu_dec = 1/10;

txtml = sprintf(' complex ML activation:%s layers:[%s]',params.layersFcn,num2str(params.hiddenSize));
cnet = complexnet(params);
cnet = cnet.train(nttpast,out);
outhat = cnet.test(nttpast);
outhat1 = cnet.test(nttpast1);

% matlab predictor
net = feedforwardnet(params.hiddenSize);
net = train(net,nttpast,out);
outri = net(nttpast);

% on the training data
figure(1233); clf;
plot(out-at(traininds),'g.-'); hold on;
plot(outhat,'r.-');
plot(outri,'k.-')
xlim([1 100]);

% on the test data
figure(1234); clf;
legtxt={};
plot(real(out1-at(testinds)),'.-','MarkerSize',24,'LineWidth',2);
legtxt{end+1}='Volterra process - a(0)';
hold on;
plot(real(outhat1),'o-','MarkerSize',12,'LineWidth',2);
legtxt{end+1} = 'complex ML output';
plot(real(outlinear1),'o','MarkerSize',12,'LineWidth',2)
legtxt{end+1} = 'linear output';
plot(real(outri1),'.-','MarkerSize',12,'LineWidth',2)
legtxt{end+1} = 're/im ML output';

legend(legtxt,'FontSize',24,'FontWeight','bold');
xlim([1 200]);
xlabel('sample number','FontSize',24,'FontWeight','bold');
ylabel('Real part of samples','FontSize',24,'FontWeight','bold');
title(sprintf('%s\n %s',txtvt,txtml),'FontSize',24,'FontWeight','bold');
ax=gca; ax.FontSize=16;
grid minor; grid;
boldify;

fprintf('mse outhat1 %f outri %f outlinear %f\n',...
    mean(abs(out1-outhat1).^2), ...
    mean(abs(out1-outri).^2), ...
    mean(abs(out1-outlinear1).^2));

fprintf('mse (less at) outhat1 %f outri %f outlinear %f\n',...
    mean(abs(out1-at(testinds)-outhat1).^2), ...
    mean(abs(out1-at(testinds)-outri).^2), ...
    mean(abs(out1-at(testinds)-outlinear1).^2));
