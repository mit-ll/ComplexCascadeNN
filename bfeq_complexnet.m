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


% from Z.Towfic's simple mod/demod code
numbits = 1e6;
M = 4; %set modulation order for QAM
span = 10; %span of the Tx/Rx filters in Symbols
rolloff = 0.25; %rolloff of Tx/Rx filters
nSamp = 4;      %Samples/symbol
filtDelay = log2(M)*span; %filter delay in bits
lastValidSample = numbits - ceil(filtDelay);
hMod = comm.RectangularQAMModulator(M, 'BitInput',true); %QAM modulator
%hMod = qammod(M, 'BitInput',true); %QAM modulator

hTxFilter = comm.RaisedCosineTransmitFilter('RolloffFactor',rolloff, ...
    'FilterSpanInSymbols',span,'OutputSamplesPerSymbol',nSamp);


hRxFilter = comm.RaisedCosineReceiveFilter('RolloffFactor',rolloff, ...
    'FilterSpanInSymbols',span,'InputSamplesPerSymbol',nSamp, ...
    'DecimationFactor',nSamp);
hDemod = comm.RectangularQAMDemodulator(M, 'BitOutput',true);
hError = comm.ErrorRate;

%{
% example of demod and error 
data           = randi([0 1], numbits, 1);
%modSignal      = step(hMod, data);
modSignal      = qammod(data,'InputType', 'bit');
txSignal       = step(hTxFilter, modSignal);

receivedSignal = txSignal;

rxSignal       = step(hRxFilter, receivedSignal);
receivedBits    = step(hDemod, [rxSignal((span+1):end); zeros(span,1)]);
errorStats     = step(hError, data(1:lastValidSample), receivedBits(1:lastValidSample))

fprintf('BER = %5.2e\nBit Errors = %d\nBits Transmitted = %d\n',errorStats);
%}


bw = 1e6;             % (Hz) signal bandwidth
fc = 10e6;            % (Hz) center frequency
fs = nSamp*2*(fc+bw); % (Hz) sample rate

% from passband sample rate to samples per symbol
[P,Q] = rat( nSamp*bw / fs);

%---------------------------------------------
JtoSdB = 30;
snrdB = 12      % (dB) Es/No symbol-to-noise ratio

hMod.reset;
hTxFilter.reset;
data           = randi([0 1], numbits, 1);     % bits
modSignal      = step(hMod, data);             % i/q
%modSignal      = qammod(data,'InputType', 'bit');
st             = step(hTxFilter, modSignal);   % i/q at nSamp per symbol so bw*nSamp
sbb            = resample( st, Q, P);          % complex baseband at fs

hMod.reset;
hTxFilter.reset;
datai          = randi([0 1], numbits, 1);
modSignali     = 10.^(JtoSdB/10)*step(hMod, datai);
%modSignali     = 10.^(JtoSdB/10)*qammod(datai,'InputType', 'bit');
xt             = step(hTxFilter, modSignali);
ibb            = resample( xt, Q, P);

st = transpose(st);
xt = transpose(xt);
sbb = transpose(sbb);
ibb = transpose(ibb);

%---------------------------------------------
% modulate to passband fc where nonlinearity takes place
tvec = (0:length(ibb)-1)/fs; 
carriersignal = exp(1i*2*pi*fc*tvec);

% can affect the phase over frequency.  not a dominant effect
finterf = 1e3;  

trainingduration = 5e-3;

%trainingduration = 1e-3;

% based on duration of the training to determine a frequency offset
%trainingduration = 0.8e-3;

% cnet ber lower than net
%trainingduration = 0.6e-3;


%trainingduration = 0.4e-3;




fsoi = 1/10 * 1/trainingduration;

carriersignalinterf = carriersignal.*exp(1i*2*pi*finterf*tvec);
carriersignalsoi = carriersignal.*exp(1i*2*pi*fsoi*tvec);

ipass = real(ibb).*real(carriersignalinterf) - imag(ibb).*imag(carriersignalinterf);
spass = real(sbb).*real(carriersignalsoi) - imag(sbb).*imag(carriersignalsoi);
%---------------------------------------------


%---------------------------------------------
% setup the problem geometry
% 2 lambda aperture
% signals separated by some multiple ofbeamwidth
numants = 2;
c_sol = 3e8;
lambda = c_sol/fc;
%pos = [ -35.4396  -26.9747; -35.6057   50.2903; 0         0]
pos = [-1.25 1.25;0  0; 0 0]* lambda;

D = sqrt(mean(var(pos(1:2,:).')));
beamwidth = 0.891 * lambda/ D;
%beamwidth = 1./(2*snr(ss)) .* (lambda/(2*pi))^2 * inv(pos*pos');

us = [0; 0; 1];
taus = us'*pos/c_sol;
vs = exp(1i*2*pi*taus(:)*fc)/sqrt(numants);

th = linspace(0,2*pi,100);
r = linspace(0,1,1000);
[rr,theta] = meshgrid(r,th); rr = transpose(rr(:)); theta = transpose(theta(:));
ux = rr.*cos(theta); uy = rr.*sin(theta);
uz = 1 - sqrt(ux.^2 + uy.^2);
uu = [ux; uy; uz];
v = exp(1i*2*pi*pos'*uu/lambda)/sqrt(numants);
beamsum = 20*log10(abs(vs'*v));
[vsAz, vsEl] = xyztodoa([1;0;0],[0;1;0],[0;0;1],uu); vsEl=pi/2-vsEl;

%indtouse = randi(numel(vsEl));
[~,indtouse] = min(abs(beamsum - -10));

iel = vsEl( indtouse);
iaz = vsAz( indtouse);

ui = [cos(iaz)*sin(iel); sin(iaz)*sin(iel); cos(iel)];
taui = ui'*pos/c_sol;
vi = exp(1i*2*pi*taui(:)*fc)/sqrt(numants);

fprintf('iaz %0.2f, iel %0.2f, separation in angle %0.2f, dot product %0.2f (db)\n',...
    iaz*180/pi,iel*180/pi,acosd( abs(vi'*vs) ),20*log10(abs(vi'*vs)));


figure(188);clf; scatter(ux,uy,20,beamsum,'filled'); caxis([-5 0]);
hold on;plot(ux(indtouse),uy(indtouse),'rx','MarkerSize',30,'LineWidth',4);
xlabel('u_x'); ylabel('u_y'); title('antenna beamsum');
hc=colorbar; ylabel(hc,'beamsum (dB)');
slidify;

%figure(189);clf; scatter(ux,uy,20,vsEl*180/pi,'filled'); colorbar;
figure(190); plot(pos(1,:)/lambda,pos(2,:)/lambda,'k.','MarkerSize',24); axis([-2 2 -2 2]);

%---------------------------------------------

%---------------------------------------------
% this filter is applied to the passband signal->modulated to baseband
% lowpass filer designed to cut off at bw compared to fs
lpFilt = designfilt('lowpassfir', 'PassbandFrequency', bw/(fs/2),...
    'StopbandFrequency', bw/(fs/2)*1.1, 'PassbandRipple', 0.5, ...
    'StopbandAttenuation', 65, 'DesignMethod', 'kaiserwin');
[Gd,w] = grpdelay(lpFilt);
gdtouse = floor(Gd(1));
%---------------------------------------------

%%
%backoffdBs = -30:2:-1;
%backoffdBs = -30:5:-5;
backoffdBs = -15;

% apply a nonlinear function to zpass (e.g. clipping) -> znl
%nlChoice = 'volterra';
nlChoice = 'tanh';'clip';'linear';
switch nlChoice
    case 'clip'
        clipleveldB = -3; %-20
        numlagsnet=4
    case 'volterra'
        %beta = [ 0.3 + 0.7*1i, 0.7 - 0.3*1i]; lags = { [1,2] , [1,3] };
        beta = [ 1, 0.3];
        lags = { [0], [1,2] };
        
        beta = [ -0.78 -1.48, 1.39, 0.04, 0.54, 3.72, 1.86, -0.76,...
            -1.62, 0.76, -0.12, 1.41, -1.52, -0.13];
        lags = { [0], [1], [2], [3] [0,0], [0 1], [0 2], [0,3], ...
            [1,1], [1,2], [1,3], [2,2], [2,3], [3,3]};        
        numlagsnet = 4*3+1
    case 'tanh'
        inr = 10^((JtoSdB + snrdB)/10);
        %numlagsnet = 2*nSamp + 1; % one more than the oversampling rate
        numlagsnet = nSamp + 1;
        %numlagsnet = nSamp
    case 'linear'
        numlagsnet = 1
end

trainingratio = trainingduration * fs * P/Q / (numants * numlagsnet)
numtraining = floor(numants*numlagsnet*trainingratio)
traininds = 1:numtraining;

for bb = 1:numel(backoffdBs)
    
    inpastnl = []; inpast=[];
    for aa = 1:numants
        
        % create noise per symbol and resample to complex baseband
        len = ceil( (gdtouse-1 + length(ipass))*P/(Q*nSamp) );
        noise = 10^(-snrdB/20) * crandn(1,len);
        noisebb = resample( noise, nSamp*Q, P);             % complex baseband
        noisebb = filter(lpFilt,noisebb);
        noisebb = noisebb(gdtouse-1+(1:numel(carriersignal)));
        noisepass = real(noisebb).*real(carriersignal) - imag(noisebb).*imag(carriersignal);
        
        %antenna delay for signal, interference
        ipass_delayed = delayseq(ipass(:),taui(aa),fs);
        spass_delayed = delayseq(spass(:),taus(aa),fs);
        
        signaldropdB = 5;
        signalphasechangerads = 10 * pi/180;
        tausexcess = 1/fs * signalphasechangerads/(2*pi);
        fprintf('bb%d aa%d: dropping signal power by %0.3f dB for nontraining\n',bb,aa,signaldropdB);
        spass_delayed(numtraining+1:end) = 10^(-signaldropdB/20) * spass_delayed(numtraining+1:end);
        
        if aa==1
            fprintf('bb%d aa%d: signal extra delay %0.5f deg phase at carrier\n',bb,aa,signalphasechangerads);
            spass_delayed(numtraining+1:end) = delayseq(spass_delayed(numtraining+1:end),tausexcess,fs);
        end
        
        interfdropdB = 3;
        interfphasechangerads = 10 * pi/180;
        tausexcess = 1/fs * interfphasechangerads/(2*pi);
        fprintf('bb%d aa%d: dropping interf power by %0.3f dB for nontraining\n',bb,aa,interfdropdB);
        ipass_delayed(numtraining+1:end) = 10^(-interfdropdB/20) * ipass_delayed(numtraining+1:end);
        
        if aa==1
            fprintf('bb%d aa%d: interf extra delay %0.5f deg phase at carrier\n',bb,aa,interfphasechangerads);
            ipass_delayed(numtraining+1:end) = delayseq(ipass_delayed(numtraining+1:end),tausexcess,fs);
        end

        
        % adding up interference, signal, and noise                
        zpass = transpose( ipass_delayed + spass_delayed  + noisepass(:));
                        
        % apply a nonlinear function to zpass (e.g. clipping) -> znl
        switch nlChoice
            case 'clip'
                znl = clipfn(zpass, 10^(clipleveldB/10) * median(abs(zpass))/sqrt(2));
            case 'volterra'
                [zpasst,znl,txtvt] = volterra(zpass,lags,beta);                
                [~,inl,~] = volterra(ipass,lags,beta);
                [~,snl,~] = volterra(spass,lags,beta);
            case 'tanh'                
                backoffdB = backoffdBs(bb) + (rand(1)*2 -1)        
                backoff = 10^(backoffdB/10);
                alpha = sqrt(backoff/inr);
                snl = tanh(alpha*spass)/alpha;
                inl = tanh(alpha*ipass)/alpha;
                znl = tanh(alpha*zpass)/alpha;
            case 'linear'
                znl = zpass;
        end
                
        % nonlinear passband signal -> complex baseband
        zbbnl = znl.*real(carriersignal) - 1i*znl.*imag(carriersignal);
        zbbnl = filter(lpFilt,zbbnl);
        zbbnl = 2*zbbnl(gdtouse:end);
        
        % linear passband signal -> complex baseband
        zbbrx = zpass.*real(carriersignal) - 1i*zpass.*imag(carriersignal);
        zbbrx = filter(lpFilt,zbbrx);
        zbbrx = 2*zbbrx(gdtouse:end);

        % resample to nSamp per symbol
        ytnl = resample(zbbnl, P, Q);
        yt = resample(zbbrx, P, Q);
        
        
        % debugging
        % baseband signal going through passband conversion and back looks
        % the same as original when carrienrsignals are the same
        if aa==1
            % interference linear
            ibb1 = ipass.*real(carriersignal) - 1i*ipass.*imag(carriersignal);
            ibb1 = filter(lpFilt,ibb1);
            ibb1 = 2*ibb1(gdtouse:end);
            
            % interference nonlinear
            ibb2 = inl.*real(carriersignal) - 1i*inl.*imag(carriersignal);
            ibb2 = filter(lpFilt,ibb2);
            ibb2 = 2*ibb2(gdtouse:end);
            
            % signal linear
            sbb1 = spass.*real(carriersignal) - 1i*spass.*imag(carriersignal);
            sbb1 = filter(lpFilt,sbb1);
            sbb1 = 2*sbb1(gdtouse:end);
            
            % signal nonlinear
            sbb2 = snl.*real(carriersignal) - 1i*snl.*imag(carriersignal);
            sbb2 = filter(lpFilt,sbb2);
            sbb2 = 2*sbb2(gdtouse:end);
            
            xt1 = resample(ibb1, P,Q);
            xt2 = resample(ibb2, P,Q);
            
            % st1 has frequency offset  
            st1 = resample(sbb1, P,Q);
            st2 = resample(sbb2, P,Q);
            
            figure(1003);clf;
            subplot(211);
            plot(real(xt)); hold on; plot(real(xt1)); xlim([1 1e3]);
            plot(real(xt2));
            legend('linear','bb->pass->bb','bb->f(pass)->bb');
            title('interference');
            subplot(212);
            plot(imag(xt)); hold on; plot(imag(xt1)); xlim([1 1e3]);
            plot(imag(xt2));
            
            figure(1004);clf;
            ha(1) = subplot(211);
            plot(real(st),'k.-'); hold on; 
            plot(real(st1),'b.-');
            plot(real(st2),'go-');
            title('signal of interest');
            legend('linear','bb->pass->bb','bb->f(pass)->bb');
            ha(2) = subplot(212);
            plot(imag(st)); hold on; plot(imag(st1));
            plot(imag(st2));
            linkaxes(ha,'x');
            xlim([1 1e3]);
            xlim(length(st) + [-1000 0]);
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
        
        % f_nl(s+i) across antennas and delays
        inpastnl = [inpastnl; yttnl];
        % (s+i) across antennas and delays
        inpast = [inpast; ytt];
    end
    
    % soi and interference cut to same length as yt so they can be used as
    % input into nn training
    st = st(1:length(yt));
    xt = xt(1:length(yt));
    
    
    if bb==1, bers = struct([]); end
    

    %----------------------------------------------------------------------
    % reference signal can have the bb->pass->bb effects including
    % frequency offset and any sampling issues
    
    %checking that resampling is transparent/linear & not affecting performance
    % however, st1 would have high BER since not aligned
    warning('Using the bb->pass->bb as the reference');
    out = st1;
    
    %warning('Using the bb as the reference');
    %out = st;
    %----------------------------------------------------------------------    
            
    % comparison plots
    st_tocompare = out;
    

    intrain = inpastnl(:,traininds); in = inpastnl; disp('NONLINEAR SAMPLES');
    %intrain = inpast(:,traininds); in = inpast; disp('LINEAR SAMPLES');
    
    outtrain = out(traininds);
    
    % linear estimate
    w=(inpast(:,traininds)*inpast(:,traininds)')\ (inpast(:,traininds)*outtrain');
    outstap = w'*inpast;
    disp('stap sinr without nonlinearity (dB)');
    [bers(bb).stap,sinrstap]= estimateber(fs * P/Q,st,outstap,data,hDemod,hError,hRxFilter,fsoi);
    
    % linear estimate when nonlinear
    w=(inpastnl(:,traininds)*inpastnl(:,traininds)')\ (inpastnl(:,traininds)*outtrain');
    outstap = w'*inpastnl;
    disp('NONLINEAR stap sinr (dB)');    
    [bers(bb).stapnl,sinrstapnl]= estimateber(fs * P/Q,st,outstap,data,hDemod,hError,hRxFilter,fsoi);
            
    %{
    prompt = 'Do you want to continue y/n [y]: ';
    str = input(prompt,'s');
    if isempty(str), str = 'y'; end
    if strcmp(str,'y')==0, return; end
    %}
    
    %{
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
    disp('check sinr for st + noise 20dB down');806970
    
    10*log10( numel(st) * abs(beta)^2/ norm(tmp-beta*st).^2 )

    disp('check sinr for st + noise 20dB down vs st1');
    beta = (tmp*st1')/norm(st1)^2;
    10*log10( numel(st1) * abs(beta)^2/ norm(tmp-beta*st1).^2 )
    %}
    
    
    %--------------------------------------------------------------------
    % non-linear predictor parameters
    params = [];
    params.domap = 'gain';'complex';'reim';
    %params.hiddenSize = [16*4 4] ;
    %params.hiddenSize = [8 4 2];
    %params.hiddenSize = [16 4 2 2];
    %params.hiddenSize = [2 2 16]; 
    %params.hiddenSize = 4*[8 4];
    %params.hiddenSize = [8 4];
    %params.hiddenSize = [32];
    %params.hiddenSize = [15 7 3];  % for numlagsnet = nSamp + 1 = 5
    %params.hiddenSize = [16 5 3];  % for numlagsnet = nSamp = 4    
    params.hiddenSize = [16 8 4];
    
    %params.hiddenSize = [4 4 4 4 4];
    %params.hiddenSize = [64 4];
    %params.hiddenSize = [2 2 2 2 2];
    %params.hiddenSize = [75 1];  % for numlagsnet = nSamp = 4
        
    % matlab predictor
    %hiddenSize_reim =  round(params.hiddenSize/1) ;
    hiddenSize_reim = [16 8 4];
    
    
    %net = feedforwardnet( hiddenSize_reim  );    
    warning('using cascade feedforwardnet');
    net = cascadeforwardnet( hiddenSize_reim  );
    net.trainFcn='trainlm'; 'trainbr';
    net.trainParam.max_fail = 5000;     % default is 6, but give it chance
    net.trainParam.epochs = 5000;
    net.trainParam.showCommandLine=true;
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
        outri = net(realifyfn(in)); outri = unrealifyfn(outri);
    else
        net = train(net,intrain,outtrain);
        outri = net(in);
    end
    
    % same as net.numWeightElements
    numb = cellfun(@numel,net.b); numb = sum(numb(:));
    numiw = cellfun(@numel,net.IW); numiw = sum(numiw(:));    
    numlw = cellfun(@numel,net.LW); numlw = sum(numlw(:));    
    nbrofParametersri = numb + numiw + numlw;    
        
    [bers(bb).net,sinrreim]= estimateber(fs * P/Q,st,outri,data,hDemod,hError,hRxFilter,fsoi);
    
    params.debugPlots=0;
    params.mu = 1e-3;
    params.trainFcn = 'trainlm'; params.nbrofEpochs = 1000
    %params.trainFcn = 'trainbr'; params.nbrofEpochs = 2000;    
    %params.trainFcn = 'Adam2'; params.nbrofEpochs = 20000;
    
    if 1
        params.minbatchsize =  'split90_10';
        params.batchtype='fixed';
    else
        %params.minbatchsize =  numants*numlagsnet*30*5;
        params.minbatchsize = floor(numel(traininds)/2);
        params.batchtype='randperm';  % works way better in general, but can stop too early
    end
    
    params.nbrofEpochs = 5000;
    params.outputFcn = 'purelin';
    params.printmseinEpochs=10;
    params.performancePlots=1;
    params.mu_inc = 10;
    params.mu_dec = 1/100;
    params.setTeacherError = 1e-6;    
    params.setTeacherEpochFrequency = 10;    
    params.max_fail = 5000;
    %--------------------------------------------------------------------

    if 1 %any(imag(intrain(:)))
        params.initFcn =  'crandn';% do not use 'c-nguyen-widrow' for complex
        params.layersFcn = 'sigrealimag2';'purelin';'cartrelu';'satlins';'myasinh';
        %params.layersFcn = {'purelin','satlins'};  %'cartrelu';'myasinh';'purelin';'satlins';'sigrealimag2';
        % advantage of cartrelu and satlins is no gain scaling
        %params.layersFcn = params.layersFcn{1:length(params.hiddenSize)};        
            
        %cnet = complexnet(params);
        params.layerConnections = 'all';
        params.inputConnections = 'all';
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
        cnet = cnet.train(intrain,outtrain); outhat = cnet.test(in);        
        %cnet = cnet.train(realifyfn(intrain),outtrain); outhat = cnet.test(realifyfn(in)); 
        %cnet = cnet.train((intrain),realifyfn(outtrain)); outhat = unrealifyfn(cnet.test(in));      
        %disp('re in re out');
        %cnet = cnet.train(realifyfn(intrain),realifyfn(outtrain)); outhat = unrealifyfn(cnet.test(realifyfn(in))); 
        %cnet = cnet.train(intrain,[outtrain; conj(outtrain)]);        
        
    else
        %params.trainFcn = 'trainbr'; params.initFcn = 'randn'
        params.trainFcn = 'trainlm'; params.initFcn = 'nguyen-widrow';        
        params.layersFcn = 'mytansig';
        %cnet = complexnet(params);

        params.layerConnections = 'all';
        params.inputConnections = 'all';
        cnet = complexcascadenet(params);
        cnet = cnet.train(realifyfn(intrain),realifyfn(outtrain));
        outhat = cnet.test(realifyfn(in)); outhat = unrealifyfn(outhat );
    end
    
    
    if iscell(params.layersFcn)
        lFcn = params.layersFcn{1};
    else
        lFcn = params.layersFcn;
    end    
    txtml = sprintf('complex ML activation:%s layers:[%s]',...
        lFcn,num2str(params.hiddenSize));

    [bers(bb).cnet,sinrc]= estimateber(fs * P/Q,st,outhat,data,hDemod,hError,hRxFilter,fsoi);    
    
    bers(bb)
        
    disp('complex sinr (dB)');
    sinrc

    disp('re/im sinr (dB)');
    sinrreim 
    
    disp('stap sinr (dB)');
    sinrstap
    
    % linear estimate when nonlinear
    disp('stap sinr when nonlinear (dB)');
    sinrstapnl
    
    figure(1233); clf;
    ha(1) = subplot(211);
    plot(real(st),'g.-','MarkerSize',20); hold on;
    plot(real(outhat-0*out),'r.-','MarkerSize',8);
    plot(real(outri-0*out),'k.-')
    plot(real(outstap-0*out),'b+-','MarkerSize',4)
    ylim(max(abs(real(st)))*[-2 2]);    
    title(sprintf('recovering s(t) from volterra( s(t) + i(t) ) J/S=%0.1fdB using split re/im tansig 8-4-1 nets',JtoSdB));
    legend('s(t)','complex network','re/im network','nl stap')
    xlabel('samples');
    ylabel('real part');
    grid minor;
    
    ha(2) = subplot(212);
    plot(imag(st),'g.-','MarkerSize',20); hold on;
    plot(imag(outhat-0*out),'r.-','MarkerSize',8);
    plot(imag(outri-0*out),'k.-')
    plot(imag(outstap-0*out),'b+-','MarkerSize',4)
    ylim(max(abs(imag(st)))*[-2 2]);
    xlabel('samples');
    ylabel('imag part');
    grid minor;
    linkaxes(ha,'x');
    %xlim(traininds(end)+[1 200]);
    xlim( numel(st) + [-5000 0]);    
    drawnow;
    pause(1);    
end


params.hiddenSize
hiddenSize_reim

return;

%%
legtxt = [];
figure(1021);clf;
tmpber=nan(1,length(backoffdBs));
for aa=1:length(backoffdBs), val = bers(aa).stap(1); if ~isempty(val),bb(aa)=val; end; end
semilogy(backoffdBs,tmpber,'.:','MarkerSize',48); hold on;
legtxt{end+1} = 'linear input, 2x4-time taps';

tmpber=nan(1,length(backoffdBs));
for aa=1:length(backoffdBs), val = bers(aa).stapnl(1); if ~isempty(val),tmpber(aa)=val; end; end
semilogy(backoffdBs,tmpber,'.:','MarkerSize',48); hold on;
legtxt{end+1} = 'nonlinear input';


tmpber=nan(1,length(backoffdBs));
for aa=1:length(backoffdBs), val = bers(aa).net(1); if ~isempty(val),tmpber(aa)=val; end; end
semilogy(backoffdBs,tmpber,'.-','MarkerSize',48);
if exist('hiddenSize_reim','var')
    netname = sprintf('%s-1 %s',num2str(hiddenSize_reim),'tansig');
else
    netname = sprintf('%s-1 %s',num2str(params.hiddenSize),'tansig');
end
legtxt{end+1} = ['re/im ' netname];

tmpber=nan(1,length(backoffdBs));
for aa=1:length(backoffdBs), val = bers(aa).cnet(1); if ~isempty(val),tmpber(aa)=val; end; end
semilogy(backoffdBs,tmpber,'.-','MarkerSize',48);
netname = sprintf('%s-1 split-%s',num2str(params.hiddenSize),'tansig');
legtxt{end+1} =['complex ' netname];


xlabel('backoff (dB)'); ylabel('Bit Error Rate (BER)');
grid minor; grid;

legend(legtxt,'Location','northwest');
title(sprintf('QPSK signal in interference\nRx compression'));
slidify;

ylim([1e-4 1]);







return;


%% alpha = backoff/inr

figure(199); clf;

JtoSdB = 30;
snrdB = 10;

inrdB = (JtoSdB + snrdB);
backoffsdB = [-20 -15 -10 -5];

legtxt={};
xx = 10.^((0:1:inrdB+30)/20);
plot(20*log10(xx),20*log10(xx),'.-'); hold on;
legtxt{end+1} = 'linear';

inr = 10.^(inrdB/10);
backoffs = 10.^(backoffsdB/10);
alpha2s = backoffs./inr;

for aa=1:numel(alpha2s)
    alpha = sqrt( alpha2s(aa) );
    % xx - alpha^2 * xx^3 /3
    plot( 20*log10(xx),20*log10(abs(tanh(alpha *xx)/alpha)),'.-');
    
    plot( 20*log10(xx),20*log10( abs(xx - 1/3 * alpha^2 * xx.^3)),'o-');
    
    legtxt{end+1} = sprintf('alpha=%f',alpha);
    legtxt{end+1} = sprintf('alpha=%f approx',alpha);
end
legend(legtxt);

%% saving some results

%{
hiddenSize = 16     8     4
hiddenSize_reim = 8     4     2
nbrofParameters = 

[420 571*2]

kk=1;
results(kk).trainingratio = 1000
results(kk).sinrc = 5.3267
results(kk).sinrreim =5.4749
results(kk).sinrstap =10.6260
results(kk).sinrstapnl =-1.6240
results(kk).ber.stap = 0.0076
results(kk).ber.stapnl = 0.4031
results(kk).ber.net = 0.0407
results(kk).ber.cnet = 0.0457
kk=kk+1
%--------------------------------------------------------------------------
results(kk).trainingratio = 2000
results(kk).sinrc = 3.3843
results(kk).sinrreim = 3.6764
results(kk).sinrstap = 10.6111
results(kk).sinrstapnl = -11.8874
results(kk).ber.stap = 0.0076
results(kk).ber.stapnl = 0.4031
results(kk).ber.net = 0.0806
results(kk).ber.cnet = 0.0766
kk=kk+1
%--------------------------------------------------------------------------
results(kk).trainingratio = 4000
results(kk).sinrc = 3.6822
results(kk).sinrreim = 3.0463
results(kk).sinrstap = 10.6197
results(kk).sinrstapnl = -11.8946
results(kk).ber.stap = 0.0075
results(kk).ber.stapnl = 0.4035
results(kk).ber.net = 0.0865
results(kk).ber.cnet = 0.0713
kk=kk+1
%--------------------------------------------------------------------------
results(kk).trainingratio = 800
results(kk).sinrc = 2.7163
results(kk).sinrreim = 3.6191
results(kk).sinrstap = 10.3911
results(kk).sinrstapnl = -7.4072
results(kk).ber.stap = 0.0139
results(kk).ber.stapnl = 0.3588
results(kk).ber.net = 0.0817
results(kk).ber.cnet = 0.1059
kk=kk+1
%--------------------------------------------------------------------------
results(kk).trainingratio = 720
results(kk).sinrc = 3.9873
results(kk).sinrreim = 4.4823
results(kk).sinrstap = 10.5731
results(kk).sinrstapnl = -7.8884
results(kk).ber.stap = 0.0089
results(kk).ber.stapnl = 0.3483
results(kk).ber.net = 0.0678
results(kk).ber.cnet = 0.0683
%--------------------------------------------------------------------------




ans =

    16     8     4


hiddenSize_reim =

    16     8     4

nbrofParametersri

nbrofParametersri =

   910

cnet.nbrofParameters*2

ans =

        1142





%--------------------------------------------------------------------------
trainingratio

trainingratio =

   800

complex sinr (dB)

sinrc =

    2.8760

re/im sinr (dB)

sinrreim =

    4.1025

stap sinr (dB)

sinrstap =

   10.6099

stap sinr when nonlinear (dB)

sinrstapnl =

   -8.0279

bers(bb).stap(1)

ans =

    0.0077

bers(bb).stapnl(1)

ans =

    0.3552

bers(bb).net(1)

ans =

    0.0707

bers(bb).cnet(1)

ans =

    0.0910
%--------------------------------------------------------------------------
trainingratio =

   600

complex sinr (dB)

sinrc =

    1.9764

re/im sinr (dB)

sinrreim =

    3.1964

stap sinr (dB)

sinrstap =

   10.5577

stap sinr when nonlinear (dB)

sinrstapnl =

  -11.2050

bers(1).stap(1)

ans =

    0.0082

bers(1).stapnl(1)

ans =

    0.4008

bers(1).net(1)

ans =

    0.0953

bers(1).cnet(1)

ans =

    0.1248


%--------------------------------------------------------------------------
trainingratio =

   400


complex sinr (dB)

sinrc =

    4.7056

re/im sinr (dB)

sinrreim =

    4.4406

stap sinr (dB)

sinrstap =

   10.5628

stap sinr when nonlinear (dB)

sinrstapnl =

    0.4071

bers(1).stap(1)

ans =

    0.0078

bers(1).stapnl(1)

ans =

    0.1712

bers(1).net(1)

ans =

    0.0637

bers(1).cnet(1)

ans =

    0.0623

%--------------------------------------------------------------------------
trainingratio =

   320


complex sinr (dB)

sinrc =

    5.0171

re/im sinr (dB)

sinrreim =

  -30.9870

stap sinr (dB)

sinrstap =

   10.6224

stap sinr when nonlinear (dB)

sinrstapnl =

    1.4015

bers(1).stap(1)

ans =

    0.0077

>> bers(1).stapnl(1)

ans =

    0.1351

>> bers(1).net(1)

ans =

    0.4923

>> bers(1).cnet(1)

ans =

    0.0574

%--------------------------------------------------------------------------
trainingratio =

   320

complex sinr (dB)

sinrc =

    0.1124

re/im sinr (dB)

sinrreim =

    1.8870

stap sinr (dB)

sinrstap =

   10.5857

stap sinr when nonlinear (dB)

sinrstapnl =

   -9.6941

bers(1).stap(1)

ans =

    0.0078

>> bers(1).stapnl(1)

ans =

    0.3879

>> bers(1).net(1)

ans =

    0.1148

>> bers(1).cnet(1)

ans =

    0.1671


%--------------------------------------------------------------------------

trainingratio =

   320


complex sinr (dB)

sinrc =

    3.4850

re/im sinr (dB)

sinrreim =

    2.2498

stap sinr (dB)

sinrstap =

   10.5593

stap sinr when nonlinear (dB)

sinrstapnl =

   -6.9380

bers(1).stap(1)

ans =

    0.0081

>> bers(1).stapnl(1)

ans =

    0.3502

>> bers(1).net(1)

ans =

    0.1100

>> bers(1).cnet(1)

ans =

    0.0873





%}