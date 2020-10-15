% some examples of use of complexnet class

addpath ./gradient-descent/


%% check for single complex layer
params =[];
params.hiddenSize=[]; 
params.outputFcn='purelin';
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

params.hiddenSize=[]; 
params.outputFcn='sigrealimag'; 
net = complexnet(params);
x=randn(1,10000)+1i*randn(1,10000);
y = net.sigrealimag(x);
[W] = net.train(x,y);
out = net.test(x);
plot(real(y),real(out),'.'); hold on;
plot(imag(y),imag(out),'o');

% input 2 x output 10 checking dimensions of w and Levenberg-Marquardt
params = [];
params.hiddenSize=[]; 
params.outputFcn='purelin';
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
params = [];
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
params = [];
params.hiddenSize=[1]; 
params.outputFcn='purelin';
params.trainFcn ='trainlm';
params.layersFcn = 'sigrealimag2';
params.nbrofEpochs=1e3;

in=randn(1,200)+1i*randn(1,200); 
out = in*(3+5*1i) + (4-2*1i);
%cnet1 = complexnet1(params);
%cnet1 = cnet1.train(in,out);

cnet = complexnet(params);
cnet = cnet.train(in,out);
print(cnet)

% purephase multiple layers
params = [];
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

params = [];
params.nbrofEpochs=200;
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


%% nonlinear volterra series
% y(t) = x(t) + sum beta * x(t-k) x(t-l)

choice = 'simple';
switch choice
    case 'simple'        
        xt = randn(1,10000) + 1i*randn(1,10000);
        L = length(xt);
        %beta = [ 0.3 + 0.7*1i, 0.7 - 0.3*1i]; lags = { [1,2] , [1,3] };
        beta = [ 1, 0.3 + 0*0.7*1i]; 
        lags = { [0], [1,2] };        
        
        numlagsnet = 6+1;
    case 'many'
        %a(t) = sqt(1-bval)*e(t) + bval*a(t-1)
        bval = 0;0.5;
        et = randn(1,100000) + 0*1i*randn(1,100000);
        if bval == 0
            xt = sqrt(1-bval^2)*et;
        else
            xt = filter(sqrt(1-bval^2),-bval,et);
        end
        beta = [ -0.78 -1.48, 1.39, 0.04, 0.54, 3.72, 1.86, -0.76,...
            -1.62, 0.76, -0.12, 1.41, -1.52, -0.13];
        lags = { [0], [1], [2], [3] [0,0], [0 1], [0 2], [0,3], ...
            [1,1], [1,2], [1,3], [2,2], [2,3], [3,3]};        
        numlagsnet = 10+1;
end

[xtt,yt,txtvt] = volterra(xt,lags,beta);


switch choice
    case 'many'
        % add measurement noise to the measured output
        snrdb = 10;
        snr = 10^(snrdb/10);
        sigma = sqrt(var(yt)) / sqrt(snr);
        yt = yt + sigma * randn(size(yt));
end

% setup the samples available for prediction (up to numlags delays)
L = length(yt);
ytt = zeros(numlagsnet,L);
for ll=0:numlagsnet-1
    ytt(ll+1, ll + (1:L-ll) ) = yt(1:L-ll);
end

traininds = 1:L/2; 
testinds=(L/2+1):L;
out = yt(traininds);
out1 = yt(testinds);

% choose the input into the network
% 1. either past output samples  (hard prediction/system id problem)
% 2. input samples               (easier)
switch choice
    case 'simple'        
        % in the simple case, can do past outputs
        inpast = ytt(2:end,traininds);
        inpast1 = ytt(2:end,testinds);
    case 'many'
        inpast = xtt(2:end,traininds);
        inpast1 = xtt(2:end,testinds);
end


% linear predictor
%betahat = conj( (inpast*inpast') \ (inpast * out') );
%outlinear = betahat'*inpast;
params = [];
params.hiddenSize = [];
params.layersFcn = 'purelin';params.outputFcn='purelin';
params.trainFcn = 'Adam2';
params.initFcn = 'nguyen-widrow';
params.nbrofEpochs = 300;
params.minbatchsize = numel(traininds)/10;
net = complexnet(params);
net = net.train(inpast,out);
outlinear1 = net.test(inpast1);

% non-linear predictor
params = [];
params.domap = 1;
params.hiddenSize = [16 6 4];
%params.hiddenSize = [16 6 ]*16;  % wrks for ~7k weights
params.debugPlots=0;
params.mu = 1e-3;
params.trainFcn = 'trainlm'; params.minbatchsize = round(numel(traininds)*0.7);
params.batchtype='fixed';
if any(imag(inpast(:)))
    params.initFcn = 'c-nguyen-widrow';
    params.layersFcn = 'sigrealimag2';'cartrelu';'satlins';
else
    params.initFcn = 'nguyen-widrow';'randn';
    params.layersFcn = 'mytansig'; %'mytanh';
end
params.outputFcn = 'purelin';
params.nbrofEpochs = 1000;
params.mu_inc = 10;
params.mu_dec = 1/10;

txtml = sprintf('complex ML activation:%s layers:[%s]',...
    params.layersFcn,num2str(params.hiddenSize));
cnet = complexnet(params);
cnet = cnet.train(inpast,out);
outhat = cnet.test(inpast);
outhat1 = cnet.test(inpast1);

% matlab predictor
net = feedforwardnet(params.hiddenSize);
if any(imag(inpast(:)))
    net = train(net,realifyfn(inpast),realifyfn(out));
    outri = net(realifyfn(inpast)); outri = outri(1:end/2,:) + 1i * outri(end/2+1:end,:);
    outri1 = net(realifyfn(inpast1)); outri1 = outri1(1:end/2,:) + 1i * outri1(end/2+1:end,:);
else
    net = train(net,inpast,out);
    outri = net(inpast);
    outri1 = net(inpast1);
end


figure(1233); clf;
if any(imag(inpast(:)))
    subplot(211)
    plot(real(out-0*xt(traininds)),'g.-','MarkerSize',20); hold on;
    plot(real(outhat),'r.-');
    plot(real(outri),'k.-')
    xlim([1 100]);
    subplot(212)
    plot(imag(out-0*xt(traininds)),'g.-','MarkerSize',20); hold on;
    plot(imag(outhat),'r.-');
    plot(imag(outri),'k.-')
    xlim([1 100]);
else
    plot(real(out-0*xt(traininds)),'g.-'); hold on;
    plot(real(outhat),'r.-');
    plot(real(outri),'k.-')
    xlim([1 100]);    
end

% on the test data
figure(1234); clf;
legtxt={};
plot(real(out1-xt(testinds)),'.-','MarkerSize',24,'LineWidth',2);
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
%boldify;

fprintf('mse outhat1 %f outri1 %f outlinear1 %f\n',...
    mean(abs(out1(:)-outhat1(:)).^2), ...
    mean(abs(out1(:)-outri1(:)).^2), ...
    mean(abs(out1(:)-outlinear1(:)).^2));

xtinds = xt(testinds);
fprintf('mse (less xt) outhat1 %f outri1 %f outlinear1 %f\n',...
    mean(abs(out1(:)-xtinds(:)-outhat1(:)).^2), ...
    mean(abs(out1(:)-xtinds(:)-outri1(:)).^2), ...
    mean(abs(out1(:)-xtinds(:)-outlinear1(:)).^2));
%%
% non-linear function approximation from Hagan & Menhaj
params = [];
params.domap = 1;
params.hiddenSize = [1 15];
params.debugPlots=0;
params.mu = 1e-3;
params.trainFcn = 'trainlm'; 
params.minbatchsize = round(numel(traininds)*0.7);
params.initFcn = 'nguyen-widrow';
params.batchtype='fixed';
params.layersFcn = 'mytansig'; %'mytanh';'sigrealimag2';
params.outputFcn = 'purelin';
params.nbrofEpochs = 300;
params.mu_inc = 10;
params.mu_dec = 1/10;

in = randn(1,1000);
out = 1/2 + 1/4 * sin(3*pi*in);

cnet = complexnet(params);
cnet = cnet.train(in,out);
outhat = cnet.test(in);
figure(101); clf;
plot(in,out,'.','MarkerSize',18); hold on;
plot(in,outhat,'o','MarkerSize',12);

cnet.test(0.1)
1/2 + 1/4 * sin(3*pi*0.1)


%%
% non-linear function approximation from Hagan & Menhaj
params = [];
params.domap = 1;
params.hiddenSize = [4 50];
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

in = 2*rand(4,100000)-1;
out = sin(2*pi*in(1,:)) .* in(2,:).^2 .* in(3,:).^3 .* in(4,:).^4 .* ...
    exp(-1*(in(1,:)+in(2,:)+in(3,:)+in(4,:)));

cnet = complexnet(params);
cnet = cnet.train(in,out);
outhat = cnet.test(in);
figure(101); clf;
plot(out,outhat,'.','MarkerSize',18);

%% clipping narrowband functions in the passband

% random noise
crandn = @(m,n) complex(randn(m,n),randn(m,n))/sqrt(2);

% some clipping nonlinearties for real passband inpput
clipfn = @(x,a) ( x.*(abs(x)<=a) + a*sign(x).*(abs(x)>a));
clip1fn = @(x) x./abs(x);
clip2fn = @(x) ( real(x).*(real(x)<=1) + real(x)>1 ) + ...
    1i* ( imag(x).*(imag(x)<=1) + imag(x)>1 );

bw = 1e6;
fc = 1e9;  % (Hz) center frequency
fs = 2*(fc+bw); % (Hz) sample rate

%---------------------------------------------
% create a random signal with given bandwidth
num=1e5;
zbb = crandn(1,num);
lpFilt = designfilt('lowpassfir', 'PassbandFrequency', bw/(fs/2),...
    'StopbandFrequency', bw/(fs/2)*1.1, 'PassbandRipple', 0.5, ...
    'StopbandAttenuation', 65, 'DesignMethod', 'kaiserwin');
[Gd,w] = grpdelay(lpFilt);
zbb=filter(lpFilt,zbb);
zbb=zbb(Gd(1):end);
%---------------------------------------------


% modulate to passband fc
tvec = (0:length(zbb)-1)/fs; s = exp(1i*2*pi*fc*tvec);
zpass = real(zbb).*real(s) - imag(zbb).*imag(s);


% apply a nonlinear function to zpass (e.g. clipping) -> znl
nlChoice = 'volterra';
switch nlChoice
    case 'clip'
        clipleveldB = -3; %-20
        znl = clipfn(zpass, 10^(clipleveldB/10) * median(abs(zpass))/sqrt(2));
        numlagsnet=2
    case 'volterra'
        %beta = [ 0.3 + 0.7*1i, 0.7 - 0.3*1i]; lags = { [1,2] , [1,3] };
        beta = [ 1, 0.3 + 0*0.7*1i];
        lags = { [0], [1,2] };        
        numlagsnet = 6+1;
        [zpasst,znl,txtvt] = volterra(zpass,lags,beta);
end

zbbnl = znl.*real(s) - 1i*znl.*imag(s);
zbbnl = filter(lpFilt,zbbnl); 
zbbnl = 2*zbbnl(Gd(1):end);

zbbrx = zpass.*real(s) - 1i*zpass.*imag(s);
zbbrx = filter(lpFilt,zbbrx); 
zbbrx = 2*zbbrx(Gd(1):end);

yt = zbbnl;

% setup the samples available for prediction (up to numlags delays)
L = length(yt);
ytt = zeros(numlagsnet,L);
for ll=0:numlagsnet-1
    ytt(ll+1, ll + (1:L-ll) ) = yt(1:L-ll);
end

traininds = 1:L/2; 
testinds=(L/2+1):L;
out = yt(traininds);
out1 = yt(testinds);

inpast = ytt(2:end,traininds);
inpast1 = ytt(2:end,testinds);

figure(131);clf;
subplot(211); 
plot(real(zbb),'.'); hold on;
plot(real(zbbrx),'.');
plot(real(zbbnl),'.');

subplot(212); 
plot(imag(zbb),'.'); hold on;
plot(imag(zbbrx),'.');
plot(imag(zbbnl),'.');

figure(132);clf;
subplot(211);
plot(abs(zbbrx),'.'); hold on;
plot(abs(zbbnl) * max(abs(zbbrx))/max(abs(zbbnl)),'.');
subplot(212);
plot(angle(zbbrx));hold on;
plot(angle(zbbnl))


% non-linear predictor
params = [];
params.domap = 1;
params.hiddenSize = [16 6 ];
%params.hiddenSize = [16 6 ]*16;  % wrks for ~7k weights
params.debugPlots=0;
params.mu = 1e-3;
params.trainFcn = 'trainlm'; params.minbatchsize = round(numel(traininds)*0.7);
params.batchtype='fixed';
if any(imag(inpast(:)))
    params.initFcn = 'c-nguyen-widrow';
    params.layersFcn = 'sigrealimag2';'cartrelu';'satlins';
else
    params.initFcn = 'nguyen-widrow';'randn';
    params.layersFcn = 'mytansig'; %'mytanh';
end
params.outputFcn = 'purelin';
params.nbrofEpochs = 1000;
params.mu_inc = 10;
params.mu_dec = 1/10;

txtml = sprintf('complex ML activation:%s layers:[%s]',...
    params.layersFcn,num2str(params.hiddenSize));

if 1
    cnet = complexnet(params);
    cnet = cnet.train(inpast,out);
    outhat = cnet.test(inpast);
    outhat1 = cnet.test(inpast1);
else
    params.initFcn = 'nguyen-widrow'
    params.layersFcn = 'mytansig'
    cnet = complexnet(params);
    cnet = cnet.train(realifyfn(inpast),realifyfn(out));
    outhat = cnet.test(realifyfn(inpast)); outhat = outhat(1:end/2,:) + 1i * outhat(end/2+1:end,:);
    outhat1 = cnet.test(realifyfn(inpast1)); outhat1 = outhat1(1:end/2,:) + 1i * outhat1(end/2+1:end,:);
end

% matlab predictor
net = feedforwardnet(params.hiddenSize);
if any(imag(inpast(:)))
    net = train(net,realifyfn(inpast),realifyfn(out));
    outri = net(realifyfn(inpast)); outri = outri(1:end/2,:) + 1i * outri(end/2+1:end,:);
    outri1 = net(realifyfn(inpast1)); outri1 = outri1(1:end/2,:) + 1i * outri1(end/2+1:end,:);
else
    net = train(net,inpast,out);
    outri = net(inpast);
    outri1 = net(inpast1);
end

figure(1233); clf;
if any(imag(inpast(:)))
    subplot(211)
    plot(real(out-0*yt(traininds)),'g.-','MarkerSize',20); hold on;
    plot(real(outhat),'r.-');
    plot(real(outri),'k.-')
    xlim([1 100]);
    subplot(212)
    plot(imag(out-0*yt(traininds)),'g.-','MarkerSize',20); hold on;
    plot(imag(outhat),'r.-');
    plot(imag(outri),'k.-')
    xlim([1 100]);
else
    plot(real(out-0*yt(traininds)),'g.-'); hold on;
    plot(real(outhat),'r.-');
    plot(real(outri),'k.-')
    xlim([1 100]);    
end



% on the test data
figure(1234); clf;
legtxt={};
plot(real(out1),'.-','MarkerSize',24,'LineWidth',2);
legtxt{end+1}='Volterra process - a(0)';
hold on;
plot(real(outhat1),'o-','MarkerSize',12,'LineWidth',2);
legtxt{end+1} = 'complex ML output';
%plot(real(outlinear1),'o','MarkerSize',12,'LineWidth',2)
%legtxt{end+1} = 'linear output';
plot(real(outri1),'.-','MarkerSize',12,'LineWidth',2)
legtxt{end+1} = 're/im ML output';

legend(legtxt,'FontSize',24,'FontWeight','bold');
%xlim([1 200]);
xlabel('sample number','FontSize',24,'FontWeight','bold');
ylabel('Real part of samples','FontSize',24,'FontWeight','bold');
title(sprintf('%s\n %s',txtvt,txtml),'FontSize',24,'FontWeight','bold');
ax=gca; ax.FontSize=16;
grid minor; grid;
%boldify;

fprintf('mse outhat1 %f outri1 %f outlinear1 %f\n',...
    mean(abs(out1(:)-outhat1(:)).^2), ...
    mean(abs(out1(:)-outri1(:)).^2), ...
    mean(abs(out1(:)-outlinear1(:)).^2));

xtinds = xt(testinds);
fprintf('mse (less xt) outhat1 %f outri1 %f outlinear1 %f\n',...
    mean(abs(out1(:)-xtinds(:)-outhat1(:)).^2), ...
    mean(abs(out1(:)-xtinds(:)-outri1(:)).^2), ...
    mean(abs(out1(:)-xtinds(:)-outlinear1(:)).^2));

%% example 12.5 from neural network design 
% for checking Jacobian calculation

in = [1 2];
out = [1 2];

% non-linear predictor
params = [];
params.domap = 0;
params.hiddenSize = [1];
params.debugPlots=1;
params.mu = 1e-3;
params.trainFcn = 'trainlm'; params.minbatchsize = length(in);
params.batchtype='fixed';
params.layersFcn{1} = 'square';
params.layersFcn{2} = 'purelin';
params.outputFcn = 'purelin';
params.initFcn = 'previous';
params.nbrofEpochs = 1;

cnet = complexnet(params);
cnet.Weights{1} = [1 0];
cnet.Weights{2} = [2 1];
cnet.debugCompare = 1;

cnet.train(in,out)

% matches with text
%Jac =
%
%     4    16
%     4     8
%     1     4
%     1     1



