function results = bfeq_sim(tx_params, chan_params, rx_params, bf_params)


n_symbols = tx_params.n_symbols;
modulationorder = tx_params.modulationorder;
span = tx_params.span;
rolloff = tx_params.rolloff;
nSamp = tx_params.nSamp;
bw = tx_params.bw;
fc = tx_params.fc;


    fs = tx_params.nSamp*2*(tx_params.fc+tx_params.bw); % (Hz) sample rate    
    % from passband sample rate to samples per symbol
    [P,Q] = rat( tx_params.nSamp*tx_params.bw / fs);    
    % from fc to fIF, set fIF to the middle of the band for best shaping
    fintermediate = fs * 2*P/Q * 1/2 * 1/2;   % (Hz) intermediate frequency

tx_params.fs = fs;
tx_params.fintermediate = fintermediate;


finterf = tx_params.finterf;
fsoi = tx_params.fsoi;

SNR_dB = chan_params.SNR_dB;
JNR_dB = chan_params.JNR_dB;
JtoS_dB = JNR_dB - SNR_dB;

interfchange_dB = chan_params.interfchange_dB;
signalchange_dB = chan_params.signalchange_dB;
signalphasechange_rads = chan_params.signalphasechange_rads;
interfphasechange_rads = chan_params.interfphasechange_rads;

backoff_dB = chan_params.backoff_dB;

n_train_symbols = bf_params.n_train_symbols;
debugPlots = bf_params.debugPlots;

% random noise interference
crandn = @(m,n) complex(randn(m,n),randn(m,n))/sqrt(2);

realifyfn = @(x) [real(x); imag(x)];
unrealifyfn = @(x) x(1:end/2,:) + 1i*x(end/2+1:end,:);

% some clipping nonlinearties for real passband input
clipfn = @(x,a) ( x.*(abs(x)<=a) + a*sign(x).*(abs(x)>a));
clip1fn = @(x) x./abs(x);
clip2fn = @(x) ( real(x).*(real(x)<=1) + real(x)>1 ) + ...
    1i* ( imag(x).*(imag(x)<=1) + imag(x)>1 );



hMod = comm.RectangularQAMModulator(modulationorder, 'BitInput',true); %QAM modulator
%hMod = qammod(M, 'BitInput',true); %QAM modulator
hTxFilter = comm.RaisedCosineTransmitFilter('RolloffFactor',rolloff, ...
    'FilterSpanInSymbols',span,'OutputSamplesPerSymbol',nSamp);

hRxFilter = comm.RaisedCosineReceiveFilter('RolloffFactor',rolloff, ...
    'FilterSpanInSymbols',span,'InputSamplesPerSymbol',nSamp, ...
    'DecimationFactor',nSamp);
hDemod = comm.RectangularQAMDemodulator(modulationorder, 'BitOutput',true);
hError = comm.ErrorRate;



hMod.reset;
hTxFilter.reset;
data           = randi([0 1], n_symbols * modulationorder, 1);  % bits
modSignal      = step(hMod, data);             % i/q
%modSignal      = qammod(data,'InputType', 'bit');
st             = step(hTxFilter, modSignal);   % i/q at nSamp per symbol so bw*nSamp
sbb            = resample( st, Q, P);          % complex baseband at fs

hMod.reset;
hTxFilter.reset;
datai          = randi([0 1], n_symbols * modulationorder, 1);
modSignali     = 10.^(JtoS_dB/10)*step(hMod, datai);
%modSignali     = 10.^(JtoS_dB/10)*qammod(datai,'InputType', 'bit');
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
carriertointermediate = exp(1i*2*pi*(fintermediate-fc)*tvec);
carriersignalinterf = carriersignal.*exp(1i*2*pi*finterf*tvec);
carriersignalsoi = carriersignal.*exp(1i*2*pi*fsoi*tvec);

ipass = real(ibb).*real(carriersignalinterf) - imag(ibb).*imag(carriersignalinterf);
spass = real(sbb).*real(carriersignalsoi) - imag(sbb).*imag(carriersignalsoi);
%---------------------------------------------


%---------------------------------------------
% setup the problem geometry
% 2 lambda aperture
% signals separated by some multiple ofbeamwidth
n_elements = 2;
c_sol = 3e8;
lambda = c_sol/fc;
%pos = [ -35.4396  -26.9747; -35.6057   50.2903; 0         0]
pos = [-1.25 1.25;0  0; 0 0]* lambda;

D = sqrt(mean(var(pos(1:2,:).')));
beamwidth = 0.891 * lambda/ D;
%beamwidth = 1./(2*snr(ss)) .* (lambda/(2*pi))^2 * inv(pos*pos');

us = [0; 0; 1];
taus = us'*pos/c_sol;
vs = exp(1i*2*pi*taus(:)*fc)/sqrt(n_elements);

th = linspace(0,2*pi,100);
r = linspace(0,1,1000);
[rr,theta] = meshgrid(r,th); rr = transpose(rr(:)); theta = transpose(theta(:));
ux = rr.*cos(theta); uy = rr.*sin(theta);
uz = 1 - sqrt(ux.^2 + uy.^2);
uu = [ux; uy; uz];
v = exp(1i*2*pi*pos'*uu/lambda)/sqrt(n_elements);
beamsum = 20*log10(abs(vs'*v));
[vsAz, vsEl] = xyztodoa([1;0;0],[0;1;0],[0;0;1],uu); vsEl=pi/2-vsEl;

%indtouse = randi(numel(vsEl));
[~,indtouse] = min(abs(beamsum - -10));

iel = vsEl( indtouse);
iaz = vsAz( indtouse);

ui = [cos(iaz)*sin(iel); sin(iaz)*sin(iel); cos(iel)];
taui = ui'*pos/c_sol;
vi = exp(1i*2*pi*taui(:)*fc)/sqrt(n_elements);


if debugPlots>1
    fprintf('iaz %0.2f, iel %0.2f, separation in angle %0.2f, dot product %0.2f (db)\n',...
        iaz*180/pi,iel*180/pi,acosd( abs(vi'*vs) ),20*log10(abs(vi'*vs)));
    
    figure(188);clf; scatter(ux,uy,20,beamsum,'filled'); caxis([-5 0]);
    hold on;plot(ux(indtouse),uy(indtouse),'rx','MarkerSize',30,'LineWidth',4);
    xlabel('u_x'); ylabel('u_y'); title('antenna beamsum');
    hc=colorbar; ylabel(hc,'beamsum (dB)');
    %slidify;
    %figure(189);clf; scatter(ux,uy,20,vsEl*180/pi,'filled'); colorbar;
    figure(190); plot(pos(1,:)/lambda,pos(2,:)/lambda,'k.','MarkerSize',24); axis([-2 2 -2 2]);
end
%---------------------------------------------

%---------------------------------------------
% this filter is applied to the passband signal->modulated to baseband
% lowpass filer designed to cut off at bw compared to fs/2
lpFilt = designfilt('lowpassfir', 'PassbandFrequency', 1.25*bw/2/(fs/2),...
    'StopbandFrequency', 1.25*bw/2/(fs/2)*1.1, 'PassbandRipple', 0.5, ...
    'StopbandAttenuation', 65, 'DesignMethod', 'kaiserwin');
[Gd,w] = grpdelay(lpFilt);
gdlowpass = floor(Gd(1));
%---------------------------------------------


%---------------------------------------------
% this filter is applied to the passband signal->modulated to the passband
% IF frequency
bpFilt = designfilt('bandpassfir', 'FilterOrder', length(lpFilt.Coefficients)-1, ...
    'CutoffFrequency1', (fintermediate-bw/2 * 1.25)/(fs/2), ...
    'CutoffFrequency2', (fintermediate+bw/2 * 1.25)/(fs/2), ...
    'StopbandAttenuation1', 60, 'PassbandRipple', 0.5, ...
    'StopbandAttenuation2', 60, 'DesignMethod', 'cls');
[Gd,w] = grpdelay(bpFilt);
gdbandpass= floor(Gd(1));
%---------------------------------------------

%---------------------------------------------
% this filter is applied to the IF passband signal->modulated to baseband
% lowpass filer designed to cut off at bw compared to fs/2 * 2*P/Q
lpFilt2 = designfilt('lowpassfir', 'PassbandFrequency', 1.25*bw/2/(fs/2 * 2*P/Q),...
    'StopbandFrequency', 1.25*bw/2/(fs/2 * 2*P/Q)*1.1, 'PassbandRipple', 0.5, ...
    'StopbandAttenuation', 65, 'DesignMethod', 'kaiserwin');
[Gd,w] = grpdelay(lpFilt2);
gdlowpass2 = floor(Gd(1));
%---------------------------------------------

if debugPlots>1
    % check that sbb and sbb--lpFilt looks about the same
    sbbtilde = filter(lpFilt,sbb);
    nfft = 2^16; fvec=(-nfft/2:nfft/2-1)/nfft * fs/2;
    figure(10); clf;
    plot(fvec/1e6, db(abs(fftshift(fft(sbb(1:nfft)))))); hold on;
    plot(fvec/1e6, db(abs(fftshift(fft(sbbtilde(1:nfft))))),'--');
end

% apply a nonlinear function to zpass (e.g. clipping) -> znl
nlChoice = 'tanh';
switch nlChoice
    case 'clip'
        clipleveldB = -3; %-20
        numlagsnet= nSamp + 1;
    case 'volterra'
        %beta = [ 0.3 + 0.7*1i, 0.7 - 0.3*1i]; lags = { [1,2] , [1,3] };
        beta = [ 1, 0.3];
        lags = { [0], [1,2] };
        
        beta = [ -0.78 -1.48, 1.39, 0.04, 0.54, 3.72, 1.86, -0.76,...
            -1.62, 0.76, -0.12, 1.41, -1.52, -0.13];
        lags = { [0], [1], [2], [3] [0,0], [0 1], [0 2], [0,3], ...
            [1,1], [1,2], [1,3], [2,2], [2,3], [3,3]};
        numlagsnet = 4*3+1;
    case 'tanh'
        inr = 10^((JtoS_dB + SNR_dB)/10);
        numlagsnet = nSamp + 1;
    case 'linear'
        numlagsnet = 1;
end

% training window at complex baseband
n_training = n_train_symbols * nSamp;
traininds = 1:n_training;

% compute equivalent window at passband, so that sensitivity
% to change can be determined
n_trainingpass = n_train_symbols * nSamp * Q/P;

inpastnl = []; inpast=[]; inpastpassnl=[];
for aa = 1:n_elements
    % create noise per symbol and resample to complex baseband
    len = ceil( (gdlowpass-1 + length(ipass))*P/(Q*nSamp) );
    noise = 10^(-SNR_dB/20) * crandn(1,len);
    noisebb = resample( noise, nSamp*Q, P);             % complex baseband
    noisebb = filter(lpFilt,noisebb);
    noisebb = noisebb(gdlowpass-1+(1:numel(carriersignal)));
    noisepass = real(noisebb).*real(carriersignal) - imag(noisebb).*imag(carriersignal);
    
    %antenna delay for signal, interference
    ipass_delayed = delayseq(ipass(:),taui(aa),fs);
    spass_delayed = delayseq(spass(:),taus(aa),fs);
    
    % sensitivity testing by changing power and phase after training
    testindspass = (n_trainingpass+1):length(spass_delayed);
    
    fprintf('aa%d: dropping signal power by %0.3f dB for nontraining\n',aa,signalchange_dB(aa));
    spass_delayed(testindspass) = 10^(-signalchange_dB(aa)/20) * spass_delayed(testindspass);
    if aa==1
        tausexcess = 1/fs * signalphasechange_rads(aa)/(2*pi);
        fprintf('aa%d: signal extra delay %0.5f deg phase at carrier\n',aa,signalphasechange_rads(aa)*180/pi);
        spass_delayed(testindspass) = delayseq(spass_delayed(testindspass),tausexcess,fs);
    end
    
    fprintf('aa%d: dropping interf power by %0.3f dB for nontraining\n',aa,interfchange_dB(aa));
    ipass_delayed(testindspass) = 10^(-interfchange_dB(aa)/20) * ipass_delayed(testindspass);
    if aa==1
        tausexcess = 1/fs * interfphasechange_rads(aa)/(2*pi);
        fprintf('aa%d: interf extra delay %0.5f deg phase at carrier\n',aa,interfphasechange_rads(aa)*180/pi);
        ipass_delayed(testindspass) = delayseq(ipass_delayed(testindspass),tausexcess,fs);
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
            backoff = 10^(backoff_dB/10);
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
    zbbnl = 2*zbbnl(gdlowpass:end);
    
    % linear passband signal -> complex baseband
    zbbrx = zpass.*real(carriersignal) - 1i*zpass.*imag(carriersignal);
    zbbrx = filter(lpFilt,zbbrx);
    zbbrx = 2*zbbrx(gdlowpass:end);
    
    % nonlinear passband signal -> passband at bw
    zpassbnl = znl.*real(carriertointermediate);
    zpassbnl = filter(bpFilt,zpassbnl);
    zpassbnl = 2*zpassbnl(gdbandpass:end);
    
    % resample to nSamp per symbol
    ytnl = resample(zbbnl, P, Q);
    ypnl = resample(zpassbnl, 2*P, Q);
    yt = resample(zbbrx, P, Q);
    
    % debugging
    % baseband signal going through passband conversion and back looks
    % the same as original when carrienrsignals are the same
    if aa==1
        % interference linear
        ibb1 = ipass.*real(carriersignal) - 1i*ipass.*imag(carriersignal);
        ibb1 = filter(lpFilt,ibb1);
        ibb1 = 2*ibb1(gdlowpass:end);
        
        % interference nonlinear
        ibb2 = inl.*real(carriersignal) - 1i*inl.*imag(carriersignal);
        ibb2 = filter(lpFilt,ibb2);
        ibb2 = 2*ibb2(gdlowpass:end);
        
        % signal linear
        sbb1 = spass.*real(carriersignal) - 1i*spass.*imag(carriersignal);
        sbb1 = filter(lpFilt,sbb1);
        sbb1 = 2*sbb1(gdlowpass:end);
        
        % passband signal -> passband at fIF
        spb1 = spass.*real(carriertointermediate);
        spb1 = filter(bpFilt,spb1);
        spb1 = 2*spb1(gdbandpass:end);
        
        % signal nonlinear
        sbb2 = snl.*real(carriersignal) - 1i*snl.*imag(carriersignal);
        sbb2 = filter(lpFilt,sbb2);
        sbb2 = 2*sbb2(gdlowpass:end);
        
        xt1 = resample(ibb1, P,Q);
        xt2 = resample(ibb2, P,Q);
        
        % st1 has frequency offset
        st1 = resample(sbb1, P,Q);
        st2 = resample(sbb2, P,Q);
        
        % passband signal with frequency offset
        sp1 = resample(spb1, 2*P,Q);
        
        if debugPlots>1
            figure(901); clf;
            d = round(length(lpFilt.Coefficients)/2);
            indstoplot = d+1+(1:1000);
            xmax = movmax(abs(spass(indstoplot)),[nSamp,nSamp]);
            plot(spass(indstoplot),'LineWidth',0.1); hold on;
            plot(xmax,'k','LineWidth',3);
            plot(-xmax,'k','LineWidth',3);
            grid minor; grid on;
            
            figure(902); clf;
            plot(real(sbb1(indstoplot(1:end/2)))); ylim([-1.2 1.2]);
            grid minor; grid on;
            
            figure(903); clf;
            plot(imag(sbb1(indstoplot(1:end/2))));  ylim([-1.2 1.2]);
            grid minor; grid on;
            
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
        
    end
    
    % f(s+i) samples
    L = length(ytnl);
    yttnl = zeros(numlagsnet,L);
    for ll=0:numlagsnet-1
        yttnl(ll+1, ll + (1:L-ll) ) = ytnl(1:L-ll);
    end
    
    % f(s+i) real passband samples
    L = length(ypnl);
    yptnl = zeros(2*numlagsnet,L);
    for ll=0:2*numlagsnet-1
        yptnl(ll+1, ll + (1:L-ll) ) = ypnl(1:L-ll);
    end
    
    % (s+i) samples
    L = length(ytnl);
    ytt = zeros(numlagsnet,L);
    for ll=0:numlagsnet-1
        ytt(ll+1, ll + (1:L-ll) ) = yt(1:L-ll);
    end
    
    % f_nl(s+i) across antennas and delays
    inpastnl = [inpastnl; yttnl];
    
    % passband f_nl(s+i) across antennas and delays
    inpastpassnl = [inpastpassnl; yptnl];
    
    % (s+i) across antennas and delays
    inpast = [inpast; ytt];
    
end

% remove the passband signals to save memory
clear zbbnl zbbrx zpassbnl;
    
% soi and interference cut to same length as yt so they can be used as
% input into nn training
st = st(1:length(yt));

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

intrain = inpastnl(:,traininds); in = inpastnl; disp('NONLINEAR SAMPLES');
%intrain = inpast(:,traininds); in = inpast; disp('LINEAR SAMPLES');

outtrain = out(traininds);


%----------------------------------------------------------------------
%       L I N E A R
%----------------------------------------------------------------------
% linear estimate when input is linear
w=(inpast(:,traininds)*inpast(:,traininds)')\ (inpast(:,traininds)*outtrain');
outstap = w'*inpast;
disp('stap sinr without nonlinearity (dB)');
[ber_stap,sinrstap]= estimateber(fs * P/Q,st,outstap,data,hDemod,hError,hRxFilter,fsoi);

% linear estimate when input is nonlinear
w=(inpastnl(:,traininds)*inpastnl(:,traininds)')\ (inpastnl(:,traininds)*outtrain');
outstap = w'*inpastnl;
disp('NONLINEAR stap sinr (dB)');
[ber_stapnl,sinrstapnl]= estimateber(fs * P/Q,st,outstap,data,hDemod,hError,hRxFilter,fsoi);
%----------------------------------------------------------------------


%----------------------------------------------------------------------
%       P A S S - B A N D
%----------------------------------------------------------------------
netpass = cascadeforwardnet( bf_params.hiddenSize_reim  );
netpass.trainFcn = bf_params.trainFcn_reim;
netpass.trainParam.max_fail = bf_params.max_fail;
netpass.trainParam.epochs = bf_params.nbrofEpochs;
netpass.trainParam.showCommandLine=true;
netpass.biasConnect = bf_params.biasConnect;
netpass = train(netpass,inpastpassnl(:,traininds),sp1(traininds));
outpass = netpass(inpastpassnl);
tvec = (0:length(outpass)-1)/ (fs * 2*P/Q);
carriertobaseband = exp(1i*2*pi*fintermediate*tvec);
outb =  outpass.*real(carriertobaseband) - 1i*outpass.*imag(carriertobaseband);
outb = filter(lpFilt2,outb);
outb = 2*outb(gdlowpass2:end);
outb = resample(outb, 1,2);
[ber_netpass,sinrpass]= estimateber(fs * P/Q,st,outb,data,hDemod,hError,hRxFilter,fsoi);
%----------------------------------------------------------------------

%----------------------------------------------------------------------
%       R E - I M
%----------------------------------------------------------------------
net = cascadeforwardnet( bf_params.hiddenSize_reim  );
net.trainFcn = bf_params.trainFcn_reim;
net.trainParam.max_fail = bf_params.max_fail;
net.trainParam.epochs = bf_params.nbrofEpochs;
net.trainParam.showCommandLine=true;
net.biasConnect = bf_params.biasConnect;
net = train(net,realifyfn(intrain),realifyfn(outtrain));
outri = net(realifyfn(in)); outri = unrealifyfn(outri);
[ber_net,sinrreim]= estimateber(fs * P/Q,st,outri,data,hDemod,hError,hRxFilter,fsoi);

% same as net.numWeightElements
numb = cellfun(@numel,net.b); numb = sum(numb(:));
numiw = cellfun(@numel,net.IW); numiw = sum(numiw(:));
numlw = cellfun(@numel,net.LW); numlw = sum(numlw(:));
nbrofParametersri = numb + numiw + numlw;
%----------------------------------------------------------------------

%----------------------------------------------------------------------
%       C O M P L E X
%----------------------------------------------------------------------
cnet = complexcascadenet(bf_params);
cnet = cnet.train(intrain,outtrain); outhat = cnet.test(in);
[ber_cnet,sinrc]= estimateber(fs * P/Q,st,outhat,data,hDemod,hError,hRxFilter,fsoi);
%----------------------------------------------------------------------

% assign results to output
results.sinrstap = sinrstap;
results.sinrstapnl = sinrstapnl;
results.sinrpass = sinrpass;
results.sinrreim = sinrreim;
results.sinrc = sinrc;
results.training_reweights = numel(traininds)/nbrofParametersri;
results.training_complexweights = numel(traininds)/cnet.nbrofParameters ;

results.ber_stap = ber_stap;
results.ber_stapnl = ber_stapnl;
results.ber_netpass = ber_netpass;
results.ber_net = ber_net;
results.ber_cnet = ber_cnet;

fprintf('\n\n--------------------------\n');
fprintf('stap sinr (dB) \t%f, ber %0.5f\n',sinrstap, results.ber_stap(1));
fprintf('stap sinr when nonlinear (dB) \t%f, ber %0.5f\n',sinrstapnl, results.ber_stapnl(1));
fprintf('pass sinr (dB) \t%f, ber %0.5f\n',sinrpass, results.ber_netpass(1));
fprintf('re/im sinr (dB) \t%f, ber %0.5f\n',sinrreim , results.ber_net(1));
fprintf('complex sinr (dB) \t%f, ber %0.5f\n',sinrc, results.ber_cnet(1));
fprintf('training/reweights %0.3f training/complexweights %0.3f\n',...
    results.training_reweights,results.training_complexweights);
fprintf('\n\n--------------------------\n');

if debugPlots>1    
    figure(1233); clf;
    ha(1) = subplot(211);
    plot(real(st),'g.-','MarkerSize',20); hold on;
    plot(real(outhat-0*out),'r.-','MarkerSize',8);
    plot(real(outri-0*out),'k.-')
    plot(real(outstap-0*out),'b+-','MarkerSize',4)
    ylim(max(abs(real(st)))*[-2 2]);
    title(sprintf('recovering s(t) from volterra( s(t) + i(t) ) J/S=%0.1fdB using split re/im tansig 8-4-1 nets',JtoS_dB));
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
end

return;
