% two -element array example

realifyfn = @(x) [real(x); imag(x)];

% random noise interference
crandn = @(m,n) complex(randn(m,n),randn(m,n))/sqrt(2);

realifyfn = @(x) [real(x); imag(x)];
unrealifyfn = @(x) x(1:end/2,:) + 1i*x(end/2+1:end,:);


% some clipping nonlinearties for real passband input
clipfn = @(x,a) ( x.*(abs(x)<=a) + a*sign(x).*(abs(x)>a));
clip1fn = @(x) x./abs(x);
clip2fn = @(x) ( real(x).*(real(x)<=1) + real(x)>1 ) + ...
    1i* ( imag(x).*(imag(x)<=1) + imag(x)>1 );

bw = 1e6;
fc = 100e6;  % (Hz) center frequency
fs = 4*2*(fc+bw); % (Hz) sample rate

[P,Q] = rat( 4*2*bw / fs);

%---------------------------------------------
% create a random signal with given bandwidth
%num=4e7;
num=4e6
JtoSdB = 30;
snrdB = 10;


ibb = crandn(1,num);
lpFilt = designfilt('lowpassfir', 'PassbandFrequency', bw/(fs/2),...
    'StopbandFrequency', bw/(fs/2)*1.1, 'PassbandRipple', 0.5, ...
    'StopbandAttenuation', 65, 'DesignMethod', 'kaiserwin');
[Gd,w] = grpdelay(lpFilt);
gdtouse = floor(Gd(1));
ibb=filter(lpFilt,ibb);
ibb=ibb(gdtouse:end);
ibb = 10^((JtoSdB+snrdB)/20) * ibb * sqrt(numel(ibb))/ norm(ibb);
%10*log10( norm(ibb)^2/numel(ibb))

% random signal
sbb = crandn(1,num);
sbb= filter(lpFilt,sbb);
sbb = sbb(gdtouse:end);
sbb = 10^(snrdB/20) * sbb * sqrt(numel(sbb))/ norm(sbb);
%10*log10( norm(sbb)^2/numel(sbb))

%oqpsk signal

%---------------------------------------------

% modulate to passband fc
tvec = (0:length(ibb)-1)/fs; 
s = exp(1i*2*pi*fc*tvec);
ipass = real(ibb).*real(s) - imag(ibb).*imag(s);
spass = real(sbb).*real(s) - imag(sbb).*imag(s);
              
xt = resample(ibb, P,Q);
st = resample(sbb, P,Q);

% setup the problem geometry
% 2 lambda aperture
% signals separated by some multiple ofbeamwidth
numants = 2;
c_sol = 3e8;
lambda = fc/c_sol;
aperture = 3 * lambda;
pos = [(2*rand(2,numants)-1); zeros(1,numants)]* aperture;
beamwidth = 0.891 * lambda/sqrt(mean(var(pos(1:2,:).'))) ;
%beamwidth = 1./(2*snr(ss)) .* (lambda/(2*pi))^2 * inv(pos*pos');
iaz = rand(1)*2*pi;
iel = beamwidth + 1/3*rand(1)*beamwidth;
us = [0; 0; 1];
ui = [cos(iaz)*sin(iel); sin(iaz)*sin(iel); cos(iel)];

taui = ui'*pos/c_sol;
taus = us'*pos/c_sol;


vi = exp(1i*2*pi*taui(:)*fc)/sqrt(numants);
vs = exp(1i*2*pi*taus(:)*fc)/sqrt(numants);


acosd( abs(vi'*vs) )


inpastnl = []; inpast=[];
for aa = 1:numants
   %antenna delay for signal, interference
       
   ipass_delayed = delayseq(ipass(:),taui(aa),fs);
   spass_delayed = delayseq(spass(:),taus(aa),fs);   
   zpass = transpose( ipass_delayed + spass_delayed );
   
   % apply a nonlinear function to zpass (e.g. clipping) -> znl
   nlChoice = 'tanh'; 'volterra';'clip';'linear';  
   
   switch nlChoice
       case 'clip'
           clipleveldB = -3; %-20
           znl = clipfn(zpass, 10^(clipleveldB/10) * median(abs(zpass))/sqrt(2));
           numlagsnet=4
       case 'volterra'
           %beta = [ 0.3 + 0.7*1i, 0.7 - 0.3*1i]; lags = { [1,2] , [1,3] };
           beta = [ 1, 0.3];
           lags = { [0], [1,2] };
           
           
           beta = [ -0.78 -1.48, 1.39, 0.04, 0.54, 3.72, 1.86, -0.76,...
               -1.62, 0.76, -0.12, 1.41, -1.52, -0.13];
           lags = { [0], [1], [2], [3] [0,0], [0 1], [0 2], [0,3], ...
               [1,1], [1,2], [1,3], [2,2], [2,3], [3,3]};
           
           
           numlagsnet = 10+1
           [zpasst,znl,txtvt] = volterra(zpass,lags,beta);
       case 'tanh'           
           backoffdB = -5; %-20 -15 -10
           inr = 10^((JtoSdB + snrdB)/10);
           backoff = 10^(backoffdB/10);
           alpha = sqrt(backoff/inr);           
           znl = tanh(alpha*zpass)/alpha;
           numlagsnet = 4*2 % one more than the oversampling rate
       case 'linear'
           znl = zpass;    
           numlagsnet = 1
   end   
   
   zbbnl = znl.*real(s) - 1i*znl.*imag(s);
   zbbnl = filter(lpFilt,zbbnl);
   zbbnl = 2*zbbnl(gdtouse:end);

   % additive receiver noise at complex baseband
   zbbnl = zbbnl + 1 * crandn(size(zbbnl,1),size(zbbnl,2));
      
   %zbbrx = ibb + sbb
   zbbrx = zpass.*real(s) - 1i*zpass.*imag(s);
   zbbrx = filter(lpFilt,zbbrx);
   zbbrx = 2*zbbrx(gdtouse:end);
   
   % additive receiver noise at complex baseband
   zbbrx = zbbrx + 1 * crandn(size(zbbrx,1),size(zbbrx,2));   

   % input is the interference signal, output is f(interference + signal)
   % see if network output - f(interference + signal) = sbb desired
   ytnl = resample(zbbnl, P, Q);
   yt = resample(zbbrx, P, Q);
   
   if aa==1
       % checking that baseband signal is same 
       ibb1 = ipass.*real(s) - 1i*ipass.*imag(s);
       ibb1 = filter(lpFilt,ibb1);
       ibb1 = 2*ibb1(gdtouse:end);
       
       sbb1 = spass.*real(s) - 1i*spass.*imag(s);
       sbb1 = filter(lpFilt,sbb1);
       sbb1 = 2*sbb1(gdtouse:end);
              
       xt1 = resample(ibb1, P,Q);
       st1 = resample(sbb1, P,Q);
       
       figure(1003);clf;
       subplot(211);
       plot(real(xt)); hold on; plot(real(xt1)); xlim([1 1e3]);
       subplot(212);
       plot(imag(xt)); hold on; plot(imag(xt1)); xlim([1 1e3]);
       drawnow;
   end
      
   % f(s+i) samples
   L = length(ytnl);
   yttnl = zeros(numlagsnet,L);
   for ll=0:numlagsnet-1
       yttnl(ll+1, ll + (1:L-ll) ) = ytnl(1:L-ll);
   end
   
   % (s+i) samples
   L = length(ytnl);
   ytt = zeros(numlagsnet,L);
   for ll=0:numlagsnet-1
       ytt(ll+1, ll + (1:L-ll) ) = yt(1:L-ll);
   end
   
   % input is f(s+i), output is s
   inpastnl = [inpastnl; yttnl];
   
   inpast = [inpast; ytt];
   
end
st = st(1:length(yt));
xt = xt(1:length(yt));


%%

warning('Using the bb->pass->bb as the reference');
out = st1;

traininds = 1:numants*numlagsnet*30; 

intrain = inpastnl(:,traininds);

outtrain = out(traininds);

% linear estimate
w=(inpast(:,traininds)*inpast(:,traininds)')\ (inpast(:,traininds)*outtrain');
outstap = w'*inpast;
disp('stap sinr (dB)');
beta = (outstap*out')/norm(out)^2;
sinr = 10*log10( numel(out) * abs(beta)^2/ norm(outstap-beta*out).^2 )

% linear estimate
w=(intrain*intrain')\ (intrain*outtrain');
outstap = w'*inpastnl;    
disp('stap nonlinear sinr (dB)');
beta = (outstap*out')/norm(out)^2;
sinrstap = 10*log10( numel(out) * abs(beta)^2/ norm(outstap-beta*out).^2 )


% check sinr for interferer
w=(inpast(:,traininds)*inpast(:,traininds)')\ (inpast(:,traininds)*xt(traininds)');
outi = w'*inpast;
disp('interferer no nl sinr (dB)');
beta = (outi*xt')/norm(xt)^2;
indstouse = 10:numel(xt)-10;
sinri = 10*log10( numel(xt) * abs(beta)^2/ norm(outi-beta*xt).^2 )
%sinri = 10*log10( numel(indstouse) * abs(beta)^2/ norm(outi(indstouse)-beta*xt(indstouse)).^2 );


tmp = st + 10^(-20/20)*crandn(1,numel(st));
beta = (tmp*st')/norm(st)^2;
disp('check sinr for st + noise 20dB down');
10*log10( numel(st) * abs(beta)^2/ norm(tmp-beta*st).^2 )

disp('check sinr for st + noise 20dB down vs st1');
beta = (tmp*st1')/norm(st1)^2;
10*log10( numel(st1) * abs(beta)^2/ norm(tmp-beta*st1).^2 )


%--------------------------------------------------------------------
% non-linear predictor parameters
params = [];
params.domap = 'reim';%'complex';
%params.hiddenSize = [16] ;
%params.hiddenSize = [8 4];
params.hiddenSize = [16 4 2 1];
%params.hiddenSize = [];

%params.hiddenSize = [16 6 ]*16;  % wrks for ~7k weights
params.debugPlots=0;
params.mu = 1e-3;
params.trainFcn = 'trainlm'; params.nbrofEpochs = 1000;
%params.trainFcn = 'Adam2'; params.nbrofEpochs = 2000;

params.minbatchsize =  'split70_30';
%params.minbatchsize =  numants*numlagsnet*30*5;

%params.batchtype='fixed';
params.batchtype='randperm';  % works way better
if any(imag(inpastnl(:))) 
    params.initFcn = 'crandn'; % do not use 'c-nguyen-widrow';
    params.layersFcn = 'sigrealimag2';'cartrelu';'satlins';
else
    params.initFcn = 'nguyen-widrow';'randn';
    params.layersFcn = 'mytansig'; %'mytanh';
end
params.outputFcn = 'purelin';
params.printmseinEpochs=10;
params.performancePlots=1;
params.mu_inc = 10;
params.mu_dec = 1/10;
%--------------------------------------------------------------------


% matlab predictor
net = feedforwardnet( round(params.hiddenSize/1 ) );
if any(imag(intrain(:)))
    DOPARTICULAR = 0;
    if DOPARTICULAR
        net = configure(net,realifyfn(intrain),realifyfn(outtrain));
        net.initFcn = 'initlay';
        [dim1,dim2] = size(net.inputWeights);
        for ii= 1:dim1
            for jj=1:dim2
                net.inputWeights{ii,jj}.initFcn = 'rands';
            end
        end
        [dim1,dim2] = size(net.inputWeights);
        for ii= 1:dim1
            for jj=1:dim2
                net.layerWeights{ii,jj}.initFcn = 'rands';
            end
        end
        [dim1] = length(net.biases);
        for ii=1:dim1, net.biases{ii}.initFcn = 'rands'; end
        net = init(net);
        
        for ii=1:length(net.layers)
            net.layers{ii}.transferFcn='satlins';
        end
    end
    net = train(net,realifyfn(intrain),realifyfn(outtrain));
    outri = net(realifyfn(inpastnl)); outri = outri(1:end/2,:) + 1i * outri(end/2+1:end,:);
else
    net = train(net,intrain,outtrain);
    outri = net(inpastnl);
end



%{
% single complex linear node = stap
params_linear = params;
params_linear.hiddenSize = [];
params_linear.trainFcn = 'trainlm'; params_linear.nbrofEpochs = 1000;
cnet_linear = complexnet(params_linear); disp('created complexnet linear, training...');
cnet_linear = cnet_linear.train(intrain,outtrain); disp('trained');
outhat_linear = cnet_linear.test(inpastnl); disp('applied on data');
%}


txtml = sprintf('complex ML activation:%s layers:[%s]',...
    params.layersFcn,num2str(params.hiddenSize));

if 1
    cnet = complexnet(params);
       
    cnet = complexcascadenet(params);
    
    %{
    % attempt to init weights, and account for mapminmax
    nbrofEpochs = cnet.nbrofEpochs;
    cnet.nbrofEpochs=1;
    cnet = cnet.train(intrain,outtrain);    
    cnet.initFcn = 'previous';
    for nn=1:cnet.nbrofLayers
        cnet.Weights{nn} = zeros( size(cnet.Weights{nn}) );
        if nn==1
            %wb = [unrealifyfn(realifyfn( conj(w)).*cnet.inputSettings.gain);0];
            wb = [conj(w);0];
            cnet.Weights{nn}(1,:) = transpose(wb);
        else
            cnet.Weights{nn}(1,1) = 1;
        end
    end
    cnet.nbrofEpochs=nbrofEpochs;
    %}

    cnet = cnet.train(intrain,outtrain);    
    
    
    
    outhat = cnet.test(inpastnl);    
else
    params.initFcn = 'nguyen-widrow'
    params.layersFcn = 'mytansig'
    cnet = complexnet(params);
    cnet = cnet.train(realifyfn(intrain),realifyfn(outtrain));
    outhat = cnet.test(realifyfn(inpastnl)); outhat = outhat(1:end/2,:) + 1i * outhat(end/2+1:end,:);
end


st_tocompare = out;

%sinr estimate
%disp('linear sinr (dB)');
%beta = (outhat_linear*st_tocompare')/norm(st_tocompare)^2;
%sinr_linear = 10*log10( numel(st_tocompare) * abs(beta)^2/ norm(outhat_linear-beta*st_tocompare).^2 )

disp('complex sinr (dB)');
beta = (outhat*st_tocompare')/norm(st_tocompare)^2;
sinrc = 10*log10( numel(st_tocompare) * abs(beta)^2/ norm(outhat-beta*st_tocompare).^2 )

disp('re/im sinr (dB)');
beta = (outri*st_tocompare')/norm(st_tocompare)^2;
sinrreim = 10*log10( numel(st_tocompare) * abs(beta)^2/ norm(outri-beta*st_tocompare).^2 )

disp('stap sinr (dB)');
beta = (outstap*st_tocompare')/norm(st_tocompare)^2;
sinrstap = 10*log10( numel(st_tocompare) * abs(beta)^2/ norm(outstap-beta*st_tocompare).^2 )

figure(1233); clf;
subplot(211)
plot(real(st),'g.-','MarkerSize',20); hold on;
plot(real(outhat-0*out),'r.-');
plot(real(outri-0*out),'k.-')
plot(real(outstap-0*out),'b+-','MarkerSize',4)
ylim(max(abs(real(st)))*[-2 2]);
xlim(traininds(end)+[1 200]);
title(sprintf('recovering s(t) from volterra( s(t) + i(t) ) J/S=%0.1fdB using split re/im tansig 8-4-1 nets',JtoSdB));
legend('random N(0,1) signal','complex network','re/im network','stap')
xlabel('samples');
ylabel('real part');
grid minor;

subplot(212)
plot(imag(st),'g.-','MarkerSize',20); hold on;
plot(imag(outhat-0*out),'r.-');
plot(imag(outri-0*out),'k.-')
plot(imag(outstap-0*out),'b+-','MarkerSize',4)
xlim(traininds(end)+[1 200]);
ylim(max(abs(imag(st)))*[-2 2]);
xlabel('samples');
ylabel('imag part');
grid minor;


%% alpha = backoff/inr

figure(199); clf;

inrdB = (JtoSdB + snrdB);
backoffsdB = [-20 -15 -10 -5 -3 -1];

legtxt={};
xx = 10.^((0:1:inrdB)/20); 
plot(20*log10(xx),20*log10(xx),'.-'); hold on; 
legtxt{end+1} = 'linear';

inr = 10.^(inrdB/10);
backoffs = 10.^(backoffsdB/10);
alpha2s = backoffs./inr;           

for aa=1:numel(alpha2s)
    alpha = sqrt( alpha2s(aa) );
    % xx - alpha^2 * xx^3 /3
    plot( 20*log10(xx),20*log10(abs(tanh(alpha *xx)/alpha)),'.-');
    legtxt{end+1} = sprintf('alpha=%f',alpha);
end
legend(legtxt);




