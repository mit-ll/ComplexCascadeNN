% two -element array example

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
nSamp = 2;
nSamp = 4      %Samples/symbol
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
fc = 10*bw;           % (Hz) center frequency
fs = nSamp*2*(fc+bw); % (Hz) sample rate

% from passband sample rate to samples per symbol
[P,Q] = rat( nSamp*bw / fs);

% from fc to fIF, set fIF to the middle of the band for best shaping
fintermediate = fs * 2*P/Q * 1/2 * 1/2;   % (Hz) intermediate frequency

%---------------------------------------------
JtoSdB = 30;
snrdB = 10      % (dB) Es/No symbol-to-noise ratio

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
carriertointermediate = exp(1i*2*pi*(fintermediate-fc)*tvec);
% can affect the phase over frequency.  not a dominant effect
finterf = 1e3;  

% based on duration of the training to determine a frequency offset
%trainingduration = 20e-3;
%trainingduration = 10e-3;
%trainingduration = 5e-3;
%trainingduration = 1e-3;
%trainingduration = 0.5e-3;
%trainingduration = 0.25e-3;
trainingduration = 0.1e-3;

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


% check that sbb and sbb--lpFilt looks about the same
sbbtilde = filter(lpFilt,sbb);
nfft = 2^16; fvec=(-nfft/2:nfft/2-1)/nfft * fs/2;
figure(10); clf;
plot(fvec/1e6, db(abs(fftshift(fft(sbb(1:nfft)))))); hold on;
plot(fvec/1e6, db(abs(fftshift(fft(sbbtilde(1:nfft))))),'--');


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
    case 'linear'
        numlagsnet = 1
end

% training window at complex baseband
trainingratio = trainingduration * fs * P/Q / (numants * numlagsnet)
numtraining = floor(numants*numlagsnet*trainingratio)
traininds = 1:numtraining;

% compute equivalent window at passband, so that sensitivity 
% to change can be determined
numtrainingpass = trainingduration * fs; 

numsim=100;
%postfix='backoffs';%for bb = 1:numel(backoffdBs)
postfix='trials';
for bb = 1:numsim
    
    inpastnl = []; inpast=[]; inpastpassnl=[];
    [interfdropdB,interfphasechangerads,...
                signaldropdB,signalphasechangerads] = deal(zeros(1,numants));
    
    for aa = 1:numants        
        % create noise per symbol and resample to complex baseband
        len = ceil( (gdlowpass-1 + length(ipass))*P/(Q*nSamp) );
        noise = 10^(-snrdB/20) * crandn(1,len);
        noisebb = resample( noise, nSamp*Q, P);             % complex baseband
        noisebb = filter(lpFilt,noisebb);
        noisebb = noisebb(gdlowpass-1+(1:numel(carriersignal)));
        noisepass = real(noisebb).*real(carriersignal) - imag(noisebb).*imag(carriersignal);
        
        %antenna delay for signal, interference
        ipass_delayed = delayseq(ipass(:),taui(aa),fs);
        spass_delayed = delayseq(spass(:),taus(aa),fs);
                
        % sensitivity testing by changing power and phase after training
        testindspass = (numtrainingpass+1):length(spass_delayed);
        
        signaldropdB(aa) = 0*5*rand(1);
        fprintf('bb%d aa%d: dropping signal power by %0.3f dB for nontraining\n',bb,aa,signaldropdB(aa));
        spass_delayed(testindspass) = 10^(-signaldropdB(aa)/20) * spass_delayed(testindspass);        
        if aa==1
            signalphasechangerads(aa) = 0*10 * pi/180 * randn(1);
            tausexcess = 1/fs * signalphasechangerads(aa)/(2*pi);        
            fprintf('bb%d aa%d: signal extra delay %0.5f deg phase at carrier\n',bb,aa,signalphasechangerads(aa)*180/pi);
            spass_delayed(testindspass) = delayseq(spass_delayed(testindspass),tausexcess,fs);
        end
        
        interfdropdB(aa) = 0*3*randn(1);
        fprintf('bb%d aa%d: dropping interf power by %0.3f dB for nontraining\n',bb,aa,interfdropdB(aa));
        ipass_delayed(testindspass) = 10^(-interfdropdB(aa)/20) * ipass_delayed(testindspass);        
        if aa==1
            interfphasechangerads(aa) = 0*10 * pi/180 * randn(1);
            tausexcess = 1/fs * interfphasechangerads(aa)/(2*pi);        
            fprintf('bb%d aa%d: interf extra delay %0.5f deg phase at carrier\n',bb,aa,interfphasechangerads(aa)*180/pi);
            ipass_delayed(testindspass) = delayseq(ipass_delayed(testindspass),tausexcess,fs);
        end
        %}
        
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
                
                %backoffdB = backoffdBs(bb) + (rand(1)*2 -1) 
                backoffdB = backoffdBs(1) + (rand(1)*2 -1)                                         
                
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
        
    %if bers(bb).stap > 0.4
    %    fprintf('ber without nonlinearity %0.5f,continuing\n',bers(bb).stap(1)); 
    %    continue;
    %end
    
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
    
    
    %----------------------------------------------------------------------
    %       P A S S - B A N D
    %----------------------------------------------------------------------
    %params.initFcn = 'nguyen-widrow';        
    %params.layersFcn = 'mytansig';
    %cnet = cnet.train(inpastpassnl(:,traininds),sp1(traininds)); outhat = cnet.test(inpastpassnl);        
    netpass = cascadeforwardnet( hiddenSize_reim  );
    netpass.trainFcn='trainlm';
    netpass.trainParam.max_fail = 1000;     % default is 6, but give it chance
    netpass.trainParam.epochs = 1000;
    netpass.trainParam.showCommandLine=true;
    netpass.biasConnect = false(size(netpass.biasConnect));
    %netpass.biasConnect(1) = true;
    netpass = train(netpass,inpastpassnl(:,traininds),sp1(traininds)); 
    outpass = netpass(inpastpassnl); 
    tvec = (0:length(outpass)-1)/ (fs * 2*P/Q);
    carriertobaseband = exp(1i*2*pi*fintermediate*tvec);
    outb =  outpass.*real(carriertobaseband) - 1i*outpass.*imag(carriertobaseband);
    outb = filter(lpFilt2,outb);
    outb = 2*outb(gdlowpass2:end);
    outb = resample(outb, 1,2);
    [bers(bb).netpass,sinrpass]= estimateber(fs * P/Q,st,outb,data,hDemod,hError,hRxFilter,fsoi);
    %----------------------------------------------------------------------
    
    
    %----------------------------------------------------------------------    
    %       R E - I M
    %----------------------------------------------------------------------
    %net = feedforwardnet( hiddenSize_reim  );    
    warning('using cascade feedforwardnet');
    net = cascadeforwardnet( hiddenSize_reim  );
    net.trainFcn='trainlm'; 'trainbr';
    net.trainParam.max_fail = 1000;     % default is 6, but give it chance
    net.trainParam.epochs = 1000;
    net.trainParam.showCommandLine=true;
    net.biasConnect = false(size(net.biasConnect));
    %net.biasConnect(1) = true;    
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
    %----------------------------------------------------------------------

    %----------------------------------------------------------------------    
    %       C O M P L E X
    %----------------------------------------------------------------------
    params.debugPlots=0;
    params.mu = 1e-3;
    params.trainFcn = 'trainlm';
    %params.trainFcn = 'trainbr';
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
    params.setTeacherError = 1e-3;    
    params.setTeacherEpochFrequency = 50;    
    params.max_fail = 2000;
    
    if 1 %any(imag(intrain(:)))
        params.initFcn =  'crandn';% do not use 'c-nguyen-widrow' for complex
        params.layersFcn = 'sigrealimag2';'purelin';'cartrelu';'satlins';'myasinh';
        %params.layersFcn = {'purelin','satlins'};  %'cartrelu';'myasinh';'purelin';'satlins';'sigrealimag2';
        % advantage of cartrelu and satlins is no gain scaling
        %params.layersFcn = params.layersFcn{1:length(params.hiddenSize)};                    
        %cnet = complexnet(params);
        params.layerConnections = 'all';
        params.inputConnections = 'all';
        
        params.biasConnect = false(length(params.hiddenSize)+1,1);
        %params.biasConnect(1) = true;          
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
    %----------------------------------------------------------------------
    
    % assign results to output    
    bers(bb).sinrstap = sinrstap;
    bers(bb).sinrstapnl = sinrstapnl;
    bers(bb).sinrreim = sinrreim;
    bers(bb).sinrc = sinrc;
    bers(bb).sinrpass = sinrpass;
    
    bers(bb).params = params;
    bers(bb).trainingratio = trainingratio;
    bers(bb).trainingduration = trainingduration;
    bers(bb).fsoi = fsoi;
    bers(bb).finterf = finterf;
    bers(bb).backoffdB = backoffdB;
    bers(bb).snrdB = snrdB;
    bers(bb).JtoSdB = JtoSdB;
    bers(bb).interfdropdB=interfdropdB;
    bers(bb).interfphasechangerads=interfphasechangerads;
    bers(bb).signaldropdB=signaldropdB;
    bers(bb).signalphasechangerads=signalphasechangerads;    
    bers(bb).training_reweights = numel(traininds)/nbrofParametersri;
    bers(bb).training_complexweights = numel(traininds)/cnet.nbrofParameters ;    
    
    fprintf('\n\n--------------------------\n');
    fprintf('complex sinr (dB) \t%f, ber %0.5f\n',sinrc, bers(bb).cnet(1));
    fprintf('pass sinr (dB) \t%f, ber %0.5f\n',sinrpass, bers(bb).netpass(1));    
    fprintf('re/im sinr (dB) \t%f, ber %0.5f\n',sinrreim , bers(bb).net(1));
    fprintf('stap sinr (dB) \t%f, ber %0.5f\n',sinrstap, bers(bb).stap(1));
    fprintf('stap sinr when nonlinear (dB) \t%f, ber %0.5f\n',sinrstapnl, bers(bb).stapnl(1));
    fprintf('training/reweights %0.3f training/complexweights %0.3f\n',bers(bb).training_reweights,bers(bb).training_complexweights);
    fprintf('\n\n--------------------------\n');
    
    
    filename = sprintf('results%0.3fms_%s.mat',trainingduration/1e-3,postfix);    
    
    save(filename,'bers','numants','bw','fc','fs','pos','nlChoice',...
        'numlagsnet','nSamp','P','Q');    
      
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
fileinfo=dir('training*.mat');
filenames = {fileinfo(:).name};
inds = contains(filenames,'backoff');
filenames = filenames(~inds);

filenames = {'results5.0ms_backoffs.mat'};

filenames = {'results5.000ms_trials.mat'};
filenames = {'results1.000ms_trials.mat'};
%filenames = {'results0.500ms_trials.mat'};
filenames = {'results0.250ms_trials.mat'};
%filenames = {'results0.100ms_trials.mat'};

for ii=1:length(filenames)
    filename = filenames{ii}
    load(filename);    
    params = bers(1).params;
    
    figure(1222); clf;
    subplot(211);
    plot([bers(:).sinrc],'.-','MarkerSize',32); hold on;
    plot([bers(:).sinrreim],'.-','MarkerSize',32)
    plot([bers(:).sinrstapnl],'.-','MarkerSize',32)
    %plot([bers(:).sinrstap],'.-','MarkerSize',32)
    xlabel('trials');
    ylabel('SINR (dB)');
    legend(sprintf('re-im %s-1 net',num2str(params.hiddenSize)),...
        sprintf('complex %s-1 net',num2str(params.hiddenSize)),...
        sprintf('stap non-linear input'),'stap','Fontsize',14,'Location','SouthEast');
    grid minor;
    title(regexprep(filename,'_','training '));
    ylim([-60 10]);
    
    subplot(212);
    tmpbernet=nan(1,length(bers));
    for aa=1:length(bers), val = bers(aa).net(1); if ~isempty(val),tmpbernet(aa)=val; end; end
    tmpbercnet=nan(1,length(bers));
    for aa=1:length(bers), val = bers(aa).cnet(1); if ~isempty(val),tmpbercnet(aa)=val; end; end    
    tmpberstapnl=nan(1,length(bers));
    for aa=1:length(bers), val = bers(aa).stapnl(1); if ~isempty(val),tmpberstapnl(aa)=val; end; end    

    semilogy(tmpbernet,'.','MarkerSize',32);hold on;
    semilogy(tmpbercnet,'.','MarkerSize',32);
    semilogy(tmpberstapnl,'.-','MarkerSize',32);
    legtxt = {sprintf('re-im %s-1 net',num2str(params.hiddenSize)),...
        sprintf('complex %s-1 net',num2str(params.hiddenSize)),...
        sprintf('stap non-linear input')};
    
    if isfield(bers(1),'netpass')
        tmpberpass=nan(1,length(bers));
        for aa=1:length(bers), val = bers(aa).netpass(1); if ~isempty(val),tmpberpass(aa)=val; end; end
        semilogy(tmpberpass,'o-','MarkerSize',12,'LineWidth',2);
        netname = sprintf('%s-1 %s',num2str(params.hiddenSize),'tansig');
        legtxt{end+1} =['real passband ' netname];
    end
    
    inds = find(tmpbernet>=tmpbercnet);
    if ~isempty(inds)
        semilogy(ones(2,1)*(inds),[tmpbernet(inds); tmpbercnet(inds)],'g-','MarkerSize',32);
    end
    legend(legtxt,'Fontsize',14,'Location','SouthEast');    
    title(sprintf('complex BER lower than re-im BER in %d/%d trials',length(inds),length(tmpbernet)));
    %[nn,xx]= hist(tmpbercnet./tmpbernet,logspace(-2,2,50))
    %bar(log10(xx),nn/sum(nn));

        
    xlabel('trails');
    ylabel('Bit Error Rate (BER)');
    grid minor; grid;
    ylim([10^-3 0.5]);
    boldify;
    drawnow; pause;
end

%%

fileinfo=dir('training*.mat');
fileinfo=dir('results*.mat');

filenames = {fileinfo(:).name};
inds = contains(filenames,'backoff');
filenames = filenames(inds);

for ii=1:length(filenames)
    filename = filenames{ii}
    load(filename);
    params = bers(1).params;
    
    % get the backoff (up to the +/- few dB randomization)
    backoffdBs = zeros(size(bers));
    for aa=1:length(bers), backoffdBs(aa) = round(bers(aa).backoffdB/5)*5; end
    
    legtxt = [];
    figure(1021);clf;
    tmpber=nan(1,length(backoffdBs));
    for aa=1:length(backoffdBs), val = bers(aa).stap(1); if ~isempty(val),tmpber(aa)=val; end; end
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
        
    if isfield(bers(1),'netpass')
        tmpber=nan(1,length(backoffdBs));
        for aa=1:length(backoffdBs), val = bers(aa).netpass(1); if ~isempty(val),tmpber(aa)=val; end; end
        semilogy(backoffdBs,tmpber,'.-','MarkerSize',48);
        netname = sprintf('%s-1 %s',num2str(params.hiddenSize),'tansig');
        legtxt{end+1} =['real passband ' netname];
    end
        
    xlabel('backoff (dB)'); ylabel('Bit Error Rate (BER)');
    grid minor; grid;
    
    legend(legtxt,'Location','southeast');
    %title(sprintf('QPSK signal in interference\nRx compression'));
    slidify;
    
    ylim([1e-4 1]);
    fnamelessextension = filename(1:end-4)
    %saveas(gcf, fullfile('figs',fnamelessextension), 'png')
    pause;
end




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
