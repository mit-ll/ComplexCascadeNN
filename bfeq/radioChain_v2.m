%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This demo simulates a full radio chain that includes:
%     1) M-QAM modulator 
%     2) BCH encoder (Message length = 16, rate = 1/4 or 1/2).
%     3) Tx raised Cosine filter
%     4) AWGN channel with SNR specification.
%     5) Rx raised cosine filter
%     6) M-QAM demodulator
%     7) BCH decoder (Message length = 16, rate = 1/4 or 1/2).
%     8) Error tallying.
% Author: Zaid J. Towfic
% Date  : 8/4/2015
%%
clear;
SNR = 5; %set SNR
M = 4; %set modulation order for QAM
Trials = 1e4;

span = 10; %span of the Tx/Rx filters in Symbols
rolloff = 0.25; %rolloff of Tx/Rx filters
nSamp = 4;      %Samples/symbol
filtDelay = log2(M)*span; %filter delay in bits

K = 1024; %number of data bits to generate at each test
messageLength = 16; %message length for encoder
rate = 1/4; %rate of encoder (codeword length = messageLength/rate)

%filtDelay bits will be wrong at the end of the demod. That is
%ceil(filtDelay/(messageLength/rate)) codewords that are wrong.
%which means ceil(filtDelay/(messageLength/rate))*messageLength bits of
%the data (after decoding) that is wrong.
lastValidSample = K - ceil(filtDelay/(messageLength/rate))*messageLength;

hEnc = comm.BCHEncoder(messageLength/rate-1,messageLength,bchgenpoly(messageLength/rate-1,messageLength,[],'double'));
hMod = comm.RectangularQAMModulator(M, 'BitInput',true); %QAM modulator
hTxFilter = comm.RaisedCosineTransmitFilter('RolloffFactor',rolloff, ...
                                            'FilterSpanInSymbols',span,'OutputSamplesPerSymbol',nSamp);
hChan = comm.AWGNChannel('NoiseMethod','Signal to noise ratio (SNR)','SNR',SNR);
hRxFilter = comm.RaisedCosineReceiveFilter('RolloffFactor',rolloff, ...
                                           'FilterSpanInSymbols',span,'InputSamplesPerSymbol',nSamp, ...
                                           'DecimationFactor',nSamp);
hDemod = comm.RectangularQAMDemodulator(M, 'BitOutput',true);
hDec = comm.BCHDecoder(messageLength/rate-1,messageLength,bchgenpoly(messageLength/rate-1,messageLength,[],'double'));
hError = comm.ErrorRate;

trial_first_err = zeros(1,Trials);
for counter = 1:Trials
  data           = randi([0 1], K, 1);
  encodedData    = step(hEnc, data);
  modSignal      = step(hMod, encodedData);
  txSignal       = step(hTxFilter, modSignal);
  receivedSignal = step(hChan, txSignal);
  rxSignal       = step(hRxFilter, receivedSignal);
  demodSignal    = step(hDemod, [rxSignal((span+1):end); zeros(span,1)]);
  receivedBits   = step(hDec, demodSignal);  
  errorStats     = step(hError, data(1:lastValidSample), receivedBits(1:lastValidSample));
  ind = find(receivedBits ~= data,1);
  if ~isempty(ind)
      trial_first_err(counter) = ind;
  else
      trial_first_err(counter) = Inf;
  end
end
fprintf('SNR = %d\nBER = %5.2e\nBit Errors = %d\nBits Transmitted = %d\nPrecision = %5.2e\n',...
    SNR,errorStats,1/(lastValidSample*length(trial_first_err)));
[min(trial_first_err) lastValidSample]
