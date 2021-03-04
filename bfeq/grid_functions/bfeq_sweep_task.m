function results = bfeq_sweep_task(taskID, outputdir)
    sweeps_per_process = 100;
    SWEEP = NaN;    
    n_iters = 40;                 % iterations per parameter value
    
    % initialize random seed
    rng(taskID);
    
    % default parameters
    tx_params.n_symbols= 5e5;
    tx_params.modulationorder = 4; % set modulation order for QAM
    tx_params.span = 10;           % span of the Tx/Rx filters in Symbols
    tx_params.rolloff = 0.25;      % rolloff of Tx/Rx filters
    tx_params.nSamp = SWEEP;       % samples/symbol
    tx_params.bw = 1e6;            % (Hz) signal bandwidth
    tx_params.fc = 10*tx_params.bw;% (Hz) center frequency
    
    tx_params.finterf = 0; 
    tx_params.fsoi = 0; 
    
    chan_params.SNR_dB = 10;
    chan_params.JNR_dB = SWEEP;
    chan_params.backoff_dB = SWEEP;
    
    chan_params.interfchange_dB = [0 0];
    chan_params.signalchange_dB = [0 0];
    chan_params.signalphasechange_rads = [0 0];
    chan_params.interfphasechange_rads = [0 0];
    
    rx_params = [];
        
    bf_params.n_train_symbols = SWEEP; % 171; 
    
    bf_params.domap = 'gain';%'complex';'reim';
    bf_params.hiddenSize = [16 8 4];
    bf_params.hiddenSize_reim = [16 8 4];
    
    if numel(bf_params.hiddenSize) ~= numel(bf_params.hiddenSize_reim)
        error('For comparison, hiddenSize should be same.');
    end
    
    bf_params.trainFcn = 'trainlm';
    bf_params.trainFcn_reim = 'trainlm';        
    bf_params.nbrofEpochs = 5000;
    bf_params.max_fail = 2000;
    
    % for cascadecomplexnet
    bf_params.minbatchsize =  'split90_10';
    bf_params.batchtype='fixed';
    bf_params.debugPlots=0;
    bf_params.outputFcn = 'purelin';
    bf_params.printmseinEpochs=10;
    bf_params.performancePlots=0;
    bf_params.mu = 1e-3;
    bf_params.mu_inc = 10;
    bf_params.mu_dec = 1/100;
    bf_params.setTeacherError = 1e-3;
    bf_params.setTeacherEpochFrequency = 50;
    
    bf_params.initFcn =  'crandn';
    bf_params.layersFcn = 'sigrealimag2';
    bf_params.layerConnections = 'all';
    bf_params.inputConnections = 'all';
    
    bf_params.biasConnect = false(length(bf_params.hiddenSize)+1,1);
    bf_params.biasConnect(1) = true;
        
    %if iscell(bf_params.layersFcn), lFcn = bf_params.layersFcn{1};
    %else, lFcn = bf_params.layersFcn; end
    %txtml = sprintf('complex ML activation:%s layers:[%s]',...
    %    lFcn,num2str(bf_params.hiddenSize));
    
    % sweep parameters that are 
    sweep.JNR_dB = [40 35 30];
    sweep.nSamp = [2 4];
    sweep.n_train_symbols = [200 400 2000 10000 20000];
    sweep.backoff_dB = -30:5:-5;
    sweep.n_iter = 1:n_iters;
    
    grid = make_sweep_grid(sweep);

    sweepvars = fieldnames(grid);
    n_sweeps = numel(grid.(sweepvars{1}));

    if taskID == 0
        fprintf('%d jobs\n', ceil(n_sweeps/sweeps_per_process))
        fprintf('****************************************\n');
        return
    end
    
    sweepidcs = (taskID-1)*sweeps_per_process + (1:sweeps_per_process);
    sweepidcs = sweepidcs(sweepidcs <= n_sweeps);
    for sweepidx = sweepidcs
        try
            params = [];
            
            rx_params = rx_params  % placeholder for possible sweeps of rx
            
            tx_params.nSamp = grid.nSamp(sweepidx);
            tx_params
    
            bf_params.n_train_symbols = grid.n_train_symbols(sweepidx);
            bf_params

            chan_params.backoff_dB = grid.backoff_dB(sweepidx);
            chan_params.JNR_dB = grid.JNR_dB(sweepidx);
            chan_params
            
            params = struct('tx_params', tx_params, 'chan_params', chan_params, 'rx_params', rx_params, 'bf_params', bf_params);
            results = bfeq_sim(tx_params, chan_params, rx_params, bf_params);
             
            disp(fullfile(outputdir, sprintf('results%06d', sweepidx)))
            save(fullfile(outputdir, sprintf('results%06d', sweepidx)), 'results', 'params', 'sweep')
        catch ME
            save(fullfile(outputdir, sprintf('error%06d', sweepidx)), 'ME')
            results.ME = ME;
            save(fullfile(outputdir, sprintf('results%06d', sweepidx)), 'results', 'params', 'sweep')
        end
    end
