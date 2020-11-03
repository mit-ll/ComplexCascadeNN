%{
All of the data below (received and transmitter source data) is sampled at Fs = 8*5/6*1e6 samps/sec.

P1dB PA operation (corresponds to the 31 dB cancellation curve from the brief):
/afs/mitll/data/2239-2601/data/fromStrokkur/Experiments/30Sept2020/iqdata/C

Linear PA operation (corresponds to the 40 dB cancellation curve from the brief):
/afs/mitll/data/2239-2601/data/fromStrokkur/Experiments/01Oct2020/iqdata/C

Each of the above directories contain received data in files that are named, for example:
rx_file_30Sept2020_expC_noisegenon43dBatten_ptobatten00db_btobatten41db_PAon.dat

The thing to make note of in the naming convention is that

*_ptobatten00db_btobatten41db_* 

means that the J/S for that experiment was approximately 41 dB +/-1dB, whereas a file with the name:

*_ptobatten40db_btobatten41db_* 

means the J/S was around 0 dB +/-1 dB (the step attenuators in the test bed are not perfect 1 dB steps). 

For the source data:

This is the file that contains the IQ samples of the communications signal (which was a repeating PRBS11 sequence with one bit (a 1) appended to the end to make each repetition of the bit sequence even length 2^11. The bits were mapped to QPSK and RRC pulse shaped, symbol rate is 5/6*1e6 syms/sec):
/afs/mitll/data/2239-2601/data/SourceData/mod_carrier_test_20200408T105551_1.dat

The jammer was the truncated white Gaussian noise (T-WGN) signal that was also pulse shaped using an RRC filter (this file is about 30 seconds long, has a symbol rate of 5/6*1e6 syms/sec):
/afs/mitll/data/2239-2601/data/SourceData/TWGN_waveform/tawgn_test_20200610T140054_1.dat


The data is formatted in the typical interleaved I/Q format (float32 for each part). 
%}


datadir = '/afs/mitll/data/2239-2601/data';
rx_dir = 'fromStrokkur/Experiments/30Sept2020/iqdata/C';

%J/S 41dB
%rx_filename = 'rx_file_30Sept2020_expC_noisegenon43dBatten_ptobatten00db_btobatten41db_PAon.dat'; sync_index = 30;

%J/S 0 dB
rx_filename = 'rx_file_30Sept2020_expC_noisegenon43dBatten_ptobatten40db_btobatten41db_PAon.dat'; sync_index = 8; 


srcdir = 'SourceData/TWGN_waveform';
src_filename = 'tawgn_test_20200610T140054_1.dat';

seqdir = 'SourceData';
seq_filename = 'mod_carrier_test_20200408T105551_1.dat';

rx_file = fullfile(datadir,rx_dir,rx_filename);
src_file = fullfile(datadir,srcdir,src_filename);
seq_file = fullfile(datadir,seqdir,seq_filename);

fn = @(x) 20*log10(x);
realifyfn = @(x) [real(x); imag(x)];

r = read_interleaved_float32(rx_file,1e5,1e6);
y = read_interleaved_float32(seq_file,1e5,1e6);
%figure(1); plot(y,'.');

fs = 8*5/6*1e6; desiredfds = -10:0.1:10;
options.doexciseband = 1; options.display = 1; options.update = 0;

% non-linear predictor parameters
params = [];
params.domap = 1;
params.hiddenSize = [16 8 4 2]; % hidden layer widths
params.debugPlots=0;
params.mu = 1e-3;              % Levenberg-Marquardt step size
params.trainFcn = 'trainlm';   % use LM
params.batchtype='fixed';      % batch to use for train/test/validate
params.initFcn = 'crandn'; % do not use 'c-nguyen-widrow';
params.layersFcn = 'sigrealimag2';'cartrelu';'satlins';'satlins';
params.outputFcn = 'purelin';  % purelin is typical
params.nbrofEpochs = 200;      % training iterations
params.mu_inc = 10;            % multiplier for mu when Hessian step increases mse
params.mu_dec = 1/10;          % multiplier for mu when Hessian step decreases mse


% samples to grab at a time
len_x = 1e6;

while(1)
    offset_bof_samps = max( floor(sync_index*len_x*(9/10)), 0);
    
    xt = read_interleaved_float32(src_file,offset_bof_samps,len_x);
    numlags = len_x/2;
    lags = -numlags:numlags;
    [g,computedfds,peaks] = computexcorr(xt(:),r(:),numlags,desiredfds,fs,options);
    
    % lag is delay on second input to match first
    lag = lags(peaks.indlag(1)); % pick the lag corresponding to max peak
    rd = transpose( delayseq( r(:), lag) );
    alpha = rd(1:len_x)*xt' /norm(xt).^2;
    yhat = rd(1:len_x) - alpha*xt;
    
    numlags = length(yhat)/2;
    lags = -numlags:numlags;
    [g,computedfds,peaks] = computexcorr(yhat(:),y(:),numlags,desiredfds,fs,options);
    lag = lags(peaks.indlag(1)); % pick the lag corresponding to max peak
    yd = transpose( delayseq( y(:), lag) );
    beta = yd(1:len_x)*yhat' /norm(yhat).^2;     
    norm(beta*yhat(1:len_xt) - yd(1:len_xt))
    
        
    % setup the samples available for prediction (up to numlags delays)
    numlagsnet = 20;
    len_xt = 1e5;
    xtt = zeros(numlagsnet,len_xt);
    for ll=0:numlagsnet-1
        xtt(ll+1, ll + (1:len_xt-ll) ) = xt(1:len_xt-ll);
    end    
    inpast = xtt(:,1:len_xt); out = rd(1:len_xt);    
    
    
    % re/im net
    net = feedforwardnet(params.hiddenSize);    
    net = train(net,realifyfn(inpast),realifyfn(out));
    outri = net(realifyfn(inpast)); outri = outri(1:end/2,:) + 1i * outri(end/2+1:end,:);    
    yhatnet = out-outri;
    
    numlags = length(yhatnet)/2;
    lags = -numlags:numlags;
    [g,computedfds,peaks] = computexcorr(yhatnet(:),y(:),numlags,desiredfds,fs,options);
    lag = lags(peaks.indlag(1)); % pick the lag corresponding to max peak
    yd = transpose( delayseq( y(:), lag) );
    betanet = yd(1:len_xt)*yhatnet' /norm(yhatnet).^2;     
    norm(betanet*yhatnet - yd(1:len_xt))    
    
    % complex net
    cnet = complexnet(params); disp('created complexnet, training...');
    cnet = cnet.train(inpast,out); disp('trained');
    outhat = cnet.test(inpast); disp('applied on data');        
    yhatcnet = out-outhat;
    
    numlags = length(yhatcnet)/2;
    lags = -numlags:numlags;
    [g,computedfds,peaks] = computexcorr(yhatcnet(:),y(:),numlags,desiredfds,fs,options);
    lag = lags(peaks.indlag(1)); % pick the lag corresponding to max peak
    yd = transpose( delayseq( y(:), lag) );
    betacnet = yd(1:len_xt)*yhatcnet' /norm(yhatcnet).^2;     
    norm(betacnet*yhatcnet - yd(1:len_xt))    
        
    lentoplot = min(1000,len_x);
    
    figure(101); clf;
    ha(1)=subplot(211);
    plot(real( alpha* xt(1:lentoplot)),'.-'); hold on;
    plot(real( rd(1:lentoplot)),'-');
    legend('synced rx samples','tx interference');
    grid minor; xlabel('samples'); ylabel('real');
    ha(2)=subplot(212);
    plot(imag( alpha* xt(1:lentoplot)),'.-'); hold on;
    plot(imag( rd(1:lentoplot)),'-');
    grid minor; xlabel('samples'); ylabel('imag');
    linkaxes(ha,'x');
    xlim([1 lentoplot]);    

        
    figure(102); clf;
    ha(1)=subplot(211);
    plot(real( beta* yhat(1:lentoplot)),'.-'); hold on;
    plot(real( betacnet* yhatcnet(1:lentoplot)),'.-'); 
    plot(real( betanet* yhatnet(1:lentoplot)),'+-','MarkerSize',2);     
    plot(real( yd(1:lentoplot)),'-');
    legend('linear estimate','complex ML','re/im ML','signal of interest');
    grid minor; xlabel('samples'); ylabel('real');    
    ha(2)=subplot(212);
    plot(imag( beta* yhat(1:lentoplot)),'.-'); hold on;
    plot(imag( betacnet* yhatcnet(1:lentoplot)),'.-');
    plot(imag( betanet* yhatnet(1:lentoplot)),'+-','MarkerSize',2);    
    plot(imag( yd(1:lentoplot)),'-');    
    grid minor; xlabel('samples'); ylabel('imag');
    linkaxes(ha,'x');
    xlim([1 lentoplot]);
    
    figure(103);clf;
    plot(( beta* yhat(1:lentoplot)),'.'); hold on;
    plot(( betacnet* yhatcnet(1:lentoplot)),'.'); 
    plot(( betanet* yhatnet(1:lentoplot)),'+','MarkerSize',2);     
    plot(( yd(1:lentoplot)),'x','MarkerSize',2);
    grid minor; xlabel('real'); ylabel('imag'); 
    legend('linear estimate','complex ML','re/im ML','signal of interest');    
    
    pause;
    sync_index = sync_index + 1;    
end

