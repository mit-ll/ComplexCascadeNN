% Compute the cross correlation function between signals x and y
% over delay (lag) and Doppler (frequency)
%
% \sum_k x[k] e^{j 2\pi f_d k T_s} y^*(k - [\tau / T_s] )
%
% Inputs:
% x          (IQ samples) x 1
% y          (IQ samples) x 1
%
% numlags    (integer) samples (-numlags:numlags)=delay of (-numlags:numlags)/fs 
% desiredfds (Hz) Doppler frequency shifts to be computed
%            if dofreqshiftexact==1, evaluate at exactly these frequencies
%            otherwise, evaluate at fs/Nfft2 spacing from lowest to highest
% fs         (Hz) sample rate
%
% options    (structure)
% .docircular       =0 for linear convolution (default)
%                   =1 for circular convolution
%                   =2 DON'T DO IFFT to get back to time (added for phase-ramp geo technique)
% .dobinspacing     use FFT bin spacing instead of trying to match computedfds
%                   (slower, since each bin offset is evaluated)
% .dofreqshift      explicity evaluate at computedfds (instead of FFT shifts)
% .dofreqshiftexact explicity evaluate at desiredfds
% .doexciseband     apply a rectangular window in the FFT domain
% .keepbw           value in (0,1) for band to keep when .doexciseband=1
% .update           display input sizes and each frequency step
% .display          also show caf
%
%
% Outputs:
% g IQ values of dim(# lags x # Dopplers) correlations
% computedfds (Hz) Doppler shifts (i.e., quantized version of desiredfds)
% peaks.tau, peaks.fd (when options.display=1) are the top peaks
%
% Example:
%{
fs = 400e3; numlags = 20; desiredfds = -50:1:50; fd = 18;
taus = (-numlags:numlags)/fs; 
msgRx = randn(fs*1,1) +1i*randn(fs*1,1); 
msgRxdelayed = [zeros(10,1);msgRx];
msgRxdelayedDoppler = msgRxdelayed.*exp(1i*2*pi*fd*(0:length(msgRxdelayed)-1).'/fs);

[gsplit,computedfds] = computexcorr(msgRx,msgRxdelayedDoppler,numlags,desiredfds,fs);
gmat = transpose( cell2mat(gsplit) );
figure;
imagesc(taus*fs,computedfds,20*log10(abs(gmat)/max(abs(gmat(:)))));
title('CAF computed via FFT using computexcorr.m');
xlabel('\tau / T_s (samples)');ylabel('f_{Doppler} (Hz)');
hc = colorbar; 
ylabel(hc,'20 log_{10}| \Sigma_k x[k] e^{j 2\pi f_d k T_s} y^*(k - [\tau / T_s] ) | (dB)');
grid minor;
colormap jet;
caxis([-25 0]);
%}
%
% Author: Swaroop Appadwedula
% Version: August 2016
% Fixed bug for noncircular where length(x) had to be < length(y) 4/2019
%
% Reference:
% Tolimieri, Richard, and Shmuel Winograd. "Computing the ambiguity surface." 
% Acoustics, Speech and Signal Processing, IEEE Transactions on 33.5 (1985): 1239-1245.

function [g,computedfds,peaks] = computexcorr(x,y,numlags,desiredfds,fs,options)

if ~isvector(x), error('Input x must be a vector'); end
if ~isvector(y), error('Input y must be a vector'); end
x = x(:); y = y(:);

lenx = length(x);
leny = length(y);
len = max( lenx, leny );
Ts = 1/fs;

if (numlags>len-1)
    warning('numlags %d > max(lenx,leny)-1 %d\n',numlags,len-1); 
    numlags = len-1;
    fprintf('setting numlags = max(lenx,leny)-1 = %d\n',numlags);
end

% assumption is that larger signal is the second argument y
% if not, have to flip arguments, and flip time on the ifft output
if (numlags>leny-1) 
    warning('Have to swap x<->y inputs.  Avoid warning by y as the longer signal');
    flip = 1; tmp = y; y = x; x = tmp; clear tmp;
    lenx = length(x); leny = length(y);
else
    flip = 0; 
end

if ~exist('options','var')
    options = struct();
end
if ~isfield(options,'docircular'), options.docircular = 0; end
if ~isfield(options,'dobinspacing'), options.dobinspacing = 1; end
if ~isfield(options,'dofreqshift'), options.dofreqshift = 0; end
if ~isfield(options,'dofreqshiftexact'), options.dofreqshiftexact = 0; end    
if ~isfield(options,'doexciseband'), options.doexciseband = 0; end    
if ~isfield(options,'keepbw'), options.keepbw = 0.6; end    
if ~isfield(options,'display'), options.display = 0; end    
if ~isfield(options,'update'), options.update = 1; end

if options.dofreqshift && options.dofreqshiftexact
    disp('Both dofreqshift and dofreqshiftexact set - pick one or the other');
    return;
end

if options.update   
    fprintf('computexcorr: x:(%dx%d) y:(%dx%d) input options are \n',...
        size(x,1),size(x,2),size(y,1),size(y,2));
    disp(options);
end

switch options.docircular
    case 0
        % for liner convolution to equal circular convolution over all lags,
        % would compute 2*len FFT, but only need numlags values so shorter FFT 
        % that just accounts for these lags is sufficient
        Nfft2 = 2^( nextpow2(len+numlags) );   
    case {1,2}        
        % for circular convolution set length(x)=length(y)=power of 2
        Nfft2 = len;
end

%{
%-------------------------------------------------------------------------
% xcorr approach with explicit dot multiply for each Doppler shift
%-------------------------------------------------------------------------
% slow due to multiplies for each Doppler shift, but exact
computedfds = desiredfds;
lenfds = length(computedfds);
g = cell(1,lenfds);
tvec = 0:Ts:(len-1)*Ts; tvec = tvec(:);
for iworker = 1:lenfds
    fprintf('iworker %d/%d\n',iworker,lenfds);
    z = x.*exp(1i*2*pi*fds(iworker)*tvec);
    g{iworker} = xcorr(z,y,numlags);%,'coeff');
end
%}


%-------------------------------------------------------------------------
% use circular shift in fft domain as Doppler shift
%-------------------------------------------------------------------------
% fixed Doppler shifts: fs * nd / Nfft2
% desired Doppler shifts are quantized into FFT sample offsets

if options.dofreqshiftexact
    nds = desiredfds/fs * Nfft2;
    computedfds = desiredfds;
else
    if options.dobinspacing
        % use bin spacing as the spacing between frequencies
        ndmax = ceil( max(abs(desiredfds))/fs * Nfft2);
        nds = (-ndmax:ndmax);
        computedfds = (-ndmax:ndmax)/Nfft2 * fs;
    else
        % find the closest match in the bin spacing to the desired frequencies
        nds = round( desiredfds/fs * Nfft2);
        nds = unique(nds);
        computedfds = nds/Nfft2 * fs;
    end
end
lennds = length(nds);

if (options.dofreqshift==0 && options.dofreqshiftexact==0)
    % -----------
    % zero-padded excess-length FFT of received
    x = fft( x ,Nfft2,1);
    % -----------
else
    %warning! x is time-domain in this case
end

switch (options.docircular)
    case {0}
        % -----------
        % zero-padded excess-length FFT ( conjugated, bit reversed) of template
        %ytilde = flip(conj(y));
        y = flipdim(conj(y),1);
        y = fft( y  ,Nfft2,1);
        % -----------
    case {1,2}
        y = conj( fft( y  ,Nfft2,1));
end

%-------------------------------------------------------------------------
% zero out the ends of the band
%-------------------------------------------------------------------------
ftozero = round( Nfft2/2 * (1-options.keepbw) );
if (options.doexciseband)
    fprintf('computexcorr: Keeping %0.5f of the bandwidth\n',options.keepbw);
    y(Nfft2/2+ (-ftozero:ftozero) ) = 0;    
    if (options.dofreqshift==0 && options.dofreqshiftexact==0)
        x(Nfft2/2+ (-ftozero:ftozero) ) = 0;
    else
        %x is time-domain in this case, so have to do excision later
    end
end

%-------------------------------------------------------------------------
% obtain the correlation for each Doppler
% use cell array with the hope of parallel computing in the future; however
% currently insufficient memory for very large arrays (128M)
%-------------------------------------------------------------------------
g = cell(1,lennds);  
for iworker = 1:lennds
        
    if options.update
        fprintf('iworker %d/%d: Doppler %0.1fHz (circshift %0.1f) ifft(2^%d=%d Kpt)\n',...
            iworker,lennds,computedfds(iworker),nds(iworker),...
            log2(Nfft2),Nfft2/1024);
    end
           
    % -----------
    % frequency shift (doppler) and multiply (time-domain correlation)  
    % options: (1) circular shift or (2) sample-by-sample multiply
    if (options.dofreqshift || options.dofreqshiftexact)        
        fshift = transpose( exp(1i*2*pi*computedfds(iworker)*Ts*(0:length(x)-1)) );        
        Ztilden = fft( x.*fshift ,Nfft2,1);
        if (options.doexciseband) 
            % excise bands as close to desired as possible
            Ztilden( Nfft2/2+ (-ftozero:ftozero) + round(nds(iworker)) ) = 0;            
        end
        Ztilden = Ztilden.*y;                         
    else
        % circular shift for Doppler
        % (inplace) multiply in freq. domain
        Ztilden = circshift(x,[nds(iworker),0]).*y;                 
    end    
    % -----------        
        
    % compute all delays, pick the desired lags to keep (o.w. too large)
    % -----------
    % ifft and assign only the portion of lags that are of interest
    switch (options.docircular)
        case {0}
            Ztilden = ifft(Ztilden,Nfft2,1);                    % back to time domain   
            % fixed bug changed len -> leny SA 4/2019
            % needs another fix when numlags is large, and y is not the biggest signal SA 5/2019
            switch flip
                case 0, g{iworker} = Ztilden( leny + (-numlags:numlags) );
                case 1, g{iworker} = Ztilden( leny + (numlags:-1:-numlags) );
            end
        case {1}
            Ztilden = ifft(Ztilden,Nfft2,1);                    % back to time domain    
            Ztilden = fftshift(Ztilden,1);
            g{iworker} = Ztilden( len/2  + (-numlags:numlags) );
        case {2}
            % don't go back to time domain
            g{iworker} = Ztilden;
    end
    % -----------
    
    %{
    % compute a portion of delays via filter-downsample
    %   resampling computationally intensive (@todo: predesign filter)
    % -----------
    % ifft and assign only the portion of lags that are of interest
    Nfftlags = 2^( nextpow2(numlags) );    
    b = resample(zy,1,Nfft2/2/Nfftlags,2^10);    
    b = ifft(b,2*Nfftlags,1);                    % back to time domain
    g{iworker} = b;
    % -----------
    %}
    
    %{
    % compute a portion of delays using matrix multiply
    %   (W too big to fit in memory. duh!)
    % -----------    
    omega = 2*pi*(/fs)*(-numlags:numlags);
    omegamat = transpose(0:Nfft2-1)*omega;      
    W = exp(1i * omegamat);
    b = W * zy;
    g{iworker} = b;
    % -----------
    %}
    

    %figure(1), plot(db(abs(zy)),'color',rand(1,3)); hold on;
    %figure(2), plot(db(abs(g{iworker})),'color',rand(1,3)); hold on;
end

gmat = transpose( cell2mat(g) );
% single peak
[val,ind] = max(abs(gmat(:)));

% top nPeaks
nPeaks = 6;
bwmat = imregionalmax(abs(gmat));
bwinds = find(bwmat); [peakvals,subinds] = sort( abs(gmat(bwmat)),'descend');
inds = bwinds(subinds(1:min(nPeaks,length(subinds))));

taus = (-numlags:numlags)/fs;
[indx,indy] = ind2sub(size(gmat),ind);
[indxs,indys] = ind2sub(size(gmat),inds);

% ---------------------------------
% assign the output peaks
peaks.tau = taus(indys);
peaks.fd = computedfds(indxs);
peaks.indf = indxs;
peaks.indlag = indys;
peaks.vals = nan(1,nPeaks);
peaks.vals(1:length(inds)) = gmat(inds);
% ---------------------------------

if options.display
    fprintf('Plotting CAF image and determining peaks...\n');
    figure(3331+1); clf;
    imagesc(taus,computedfds,20*log10(abs(gmat)/val)); hold on;
    
    % top nPeaks
    plot(taus(indys),computedfds(indxs),'wx','MarkerSize',30,'LineWidth',2);
    for nn = 1:nPeaks
        text(taus(indys(nn)),computedfds(indxs(nn)),sprintf('%f\n%0.3f',taus(indys(nn)),computedfds(indxs(nn))),'color','w');
    end    
    
    % single peak
    plot(taus(indy),computedfds(indx),'mx','MarkerSize',40,'LineWidth',2);
    
    if options.update
    fprintf('computexcorr: tau (s):\n');
    taus(indys)
    fprintf('computexcorr: Doppler (Hz):\n');
    computedfds(indxs)    
    end
    
    title('CAF computed via FFT using computexcorr.m');
    xlabel('\tau (s)');ylabel('f_{Doppler} (Hz)');
    hc = colorbar;
    ylabel(hc,'20 log_{10}| \Sigma_k x[k] e^{j 2\pi f_d k T_s} y^*(k - [\tau / T_s] ) | (dB)');
    grid minor;
    colormap jet;
    caxis([-15 0]);
end

return;

%% test bench for checking accuracy of FFT-IFFT implementation
K = 2^20 - 100;  % number of samples
x1 = rand(K,1); x2 = rand(K,1);
fs = 10; numlags = 32; 
Nfft2 = 2^nextpow2(K+numlags);
fds = (-30:30)*fs/Nfft2;

disp('xcorr');
tic;
% xcorr approach
gx = zeros(numlags*2+1,length(fds));
for ii = 1:length(fds)
    gx(:,ii) = xcorr(x1.*exp(1i*2*pi*fds(ii)/fs * [0:K-1]).',x2,numlags);
end
toc;

disp('fft-ifft');
tic;
% fft-ifft approach
[g,computedfds] = computexcorr(x1,x2,numlags,fds,fs);
toc;

gc = cell2mat(g);

% compare xcorr with computexcorr using Frobenius norm of matrix difference
norm(gx-gc,'fro')^2


%% test bench for circular convolution case

M = 2;
hMod = comm.BPSKModulator('PhaseOffset',pi/M);
hDemod = comm.BPSKDemodulator('PhaseOffset',pi/M);

K = 2^15;
%K = 2^20;  % number of samples
%K = 2^26;


fsym = 57e6; 
%numlags = 32;
numlags = 50; 


options.docircular = 0;
switch options.docircular
    case 0
        % for liner convolution to equal circular convolution over all lags,
        % would compute 2*len FFT, but only need numlags values so shorter FFT is
        % sufficient
        Nfft2 = 2^( nextpow2(K+numlags) );        
    case 1        
        % for circular convolution set length(x)=length(y)=power of 2
        Nfft2 = K;        
        K = K - numlags
end

%Nfft2 = 2^nextpow2(K);
%bins = -30:30;
bins = -5:5
fds = bins*fsym/Nfft2;

choice = 'random';
switch choice
    case 'random'
        %x1 = rand(K,1) +1i*rand(K,1);
        %x2 = x1; %x2 = rand(K,1) +1i*rand(K,1);
        
        % seed for repeatability of random bits
        rChoices = {'mcg16807','mlfg6331_64','mrg32k3a','mt19937ar','shr3cong','swb2712'};
        rnum = 4;
        %hStr = RandStream(rChoices{rnum}, 'Seed', 0);
        hStr = RandStream(rChoices{rnum},'Seed','shuffle');
        
        msgData = randi(hStr,[0 M-1],K,1);    % seeded random
        %msgData = randi([0 M-1],frameLen,1);    % no seed
               
        x1 = step(hMod, msgData);        
        x2 = x1;
    case 'Chu'
        % Chu (quadratic-phase, constant amp, zero autocorrelation)
        MM = 7;
        ph = (0:K-1).^2 * MM/K;
        x1 = transpose( exp(1i*pi * ph ) );
        x2 = x1;        
    case 'PN'
        % Generator available on EBEM
        pinit = [zeros(1,22) 1]; pcoeff = [23 5 0];
        hStr = comm.PNSequence('Polynomial',pcoeff, ...
            'VariableSizeOutput', 1, 'InitialConditions',pinit,...
            'MaximumOutputSize',[K 1]);
        msgData = step(hStr,K);
        x1 = step(hMod, msgData);        
        x2 = x1;
    case 'Kasami'       
        %pinit = [zeros(1,22) 1]; pcoeff = [23 5 0];
        pinit = [zeros(1,25) 1]; pcoeff = [26 8 7 1 0];                        
        hStr = comm.KasamiSequence('Polynomial',pcoeff, ...
            'VariableSizeOutput', 1, 'InitialConditions',pinit,...
            'MaximumOutputSize',[K 1]);
        msgData = step(hStr,K);
        x1 = step(hMod, msgData);        
        x2 = x1;
end

disp('linear convolution');
tic;
gl = zeros(numlags*2+1,length(fds));
for ii = 1:length(fds)
    fprintf('linear convolution %d/%d\n',ii,length(fds));
    glval = conv(x1.*exp(1i*2*pi*fds(ii)/fsym * (0:K-1)).',conj(flipud(x2)));
    gl(:,ii) = glval( K + (-numlags:numlags) );
end
toc;
%figure; imagesc(db(abs(gl)/max(abs(gl(:))))); colorbar
%caxis([-100 0])


disp('circular convolution');
tic;
% circular convolution
gx = zeros(numlags*2+1,length(fds));
X2 = fft((x2));
for ii = 1:length(fds)
    gxval = fftshift( ifft( fft(x1.*exp(1i*2*pi*fds(ii)/fsym * (0:K-1)).').* conj(X2) ) ,1);
    gx(:,ii) = gxval( K/2 + 1 + (-numlags:numlags) );
end
toc;


disp('computexcorr');
% whether dofreqshift==1 or not, answer is the same as above
options.dofreqshift = 0; 
tic;
% fft-ifft approach
[g,computedfds] = computexcorr(x1,x2,numlags,fds,fsym,options);
toc;
gc = cell2mat(g);


% compare xcorr with computexcorr using Frobenius norm of matrix difference
disp('linear-circular'); norm(gl-gx,'fro')^2
disp('circular-computexcorr');norm(gx-gc,'fro')^2
disp('linear-computexcorr'); norm(gl-gc,'fro')^2

%inds = [1:numlags numlags+2:2*numlags+1];
%norm(gx(inds,:)-gc(inds,:),'fro')^2


figure(992); clf;
subplot(311),imagesc(-numlags:numlags,bins,db(abs(gl.')/max(abs(gl(:))))); colorbar; caxis([-100 0])
title('linear');
subplot(312),imagesc(-numlags:numlags,bins,db(abs(gx.')/max(abs(gx(:))))); colorbar; caxis([-100 0]) 
title('circular convolution');
subplot(313),imagesc(-numlags:numlags,bins,db(abs(gc.')/max(abs(gc(:))))); colorbar; caxis([-100 0])
title('computexcorr');

xlabel('lags (time)'); ylabel('bins (freq)');
subplot(311),title(sprintf('Autocorrelation delay-Doppler for BPSK %s bits',choice));

hc = colorbar; ylabel(hc,'20 log10(sum{s(n) s^*(n)})');
boldify;





