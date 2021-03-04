function [ber,sinr] = estimateber(fs,st,outhat,data,hDemod,hError,hRxFilter,fsoi,debug)

% split of integration interval as estimate of frequency resolution
fRes = 1/10 * fs/length(st);

numlags = 1000; desiredfds = fsoi + (-50:fRes:50);
taus = (-numlags:numlags)/ fs;
options.update = 0;
options.dofreqshiftexact=1;
fprintf('estimateber: computing correlation with signal waveform\n');
[gsplit,computedfds,peaks] = computexcorr(st,outhat,numlags,desiredfds,fs,options);
gmat = transpose( cell2mat(gsplit) );

if ~exist('debug','var'), debug = 0; end
if debug>1
    figure(101);
    imagesc(taus*fs,computedfds,20*log10(abs(gmat)/max(abs(gmat(:)))));
    title('CAF computed via FFT using computexcorr.m');
    xlabel('\tau / T_s (samples)');ylabel('f_{Doppler} (Hz)');
    hc = colorbar;
    ylabel(hc,'20 log_{10}| \Sigma_k x[k] e^{j 2\pi f_d k T_s} y^*(k - [\tau / T_s] ) | (dB)');
    grid minor;
    colormap jet;
    caxis([-25 0]);
end

% genie correction for frequency, since the st waveform would not be known
% for the entire time period
outcorrected = delayseq( outhat(:), peaks.tau(1) , fs) .* exp(-1i*2*pi*peaks.fd(1)*(0:length(outhat)-1).'/fs);
outcorrected = transpose( outcorrected );

% either pad or remove samples
if length(st) > length(outcorrected)
   st = st(1:length(outcorrected)); 
end
if length(outcorrected) > length(st)
    outcorrected = outcorrected(1:length(st));    
end

beta = (outcorrected*st')/norm(st)^2;
sinr = 10*log10( numel(st) * abs(beta)^2/ norm(outcorrected-beta*st).^2 );

pad = ceil(numel(outcorrected)/hRxFilter.DecimationFactor)*hRxFilter.DecimationFactor - numel(outcorrected);
receivedSignal = [outcorrected(:); zeros(pad,1)];
rxSignal       = step(hRxFilter, receivedSignal);

hDemod.reset;
hError.reset;

% have to line up the bits
span = hRxFilter.FilterSpanInSymbols;
receivedBits    = step(hDemod, [rxSignal((span+1):end); zeros(span,1)]);
newlastValidSample = length(receivedBits)-100;
errorStats     = step(hError, data(1:newlastValidSample), receivedBits(1:newlastValidSample));
fprintf('BER = %5.2e\nBit Errors = %d\nBits Transmitted = %d\n',errorStats);
ber = errorStats;

end