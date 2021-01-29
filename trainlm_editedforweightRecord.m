function [out1,out2] = trainlm(varargin)
%TRAINLM Levenberg-Marquardt backpropagation.
%
%  <a href="matlab:doc trainlm">trainlm</a> is a network training function that updates weight and
%  bias states according to Levenberg-Marquardt optimization.
%
%  <a href="matlab:doc trainlm">trainlm</a> is often the fastest backpropagation algorithm in the toolbox,
%  and is highly recommended as a first choice supervised algorithm,
%  although it does require more memory than other algorithms.
%
%  Syntax:
%       net.trainFcn = 'trainlm';
%       [net,tr] = <a href="matlab:doc train">train</a>(net,...)
%
%  Training occurs according to training parameters, listed here with their
%  default values:
%    epochs            1000  Maximum number of epochs to train
%    goal                 0  Performance goal
%    max_fail             6  Maximum validation failures
%    min_grad          1e-7  Minimum performance gradient
%    mu               0.001  Initial Mu
%    mu_dec             0.1  Mu decrease factor
%    mu_inc              10  Mu increase factor
%    mu_max            1e10  Maximum Mu
%    show                25  Epochs between displays
%    showCommandLine  false  Generate command-line output
%    showWindow        true  Show training GUI
%    time               inf  Maximum time to train in seconds
%
%  To make this the default training function for a network, and view
%  and/or change parameter settings, use these two properties:
%
%    net.<a href="matlab:doc nnproperty.net_trainFcn">trainFcn</a> = '<a href="matlab:doc trainlm">trainlm</a>';
%    net.<a href="matlab:doc nnproperty.net_trainParam">trainParam</a>
%
%  See also trainscg, feedforwardnet, narxnet.

% Copyright 1992-2017 The MathWorks, Inc.

%% =======================================================
%  BOILERPLATE_START
%  This code is the same for all Training Functions.

if nargin > 0
    [varargin{:}] = convertStringsToChars(varargin{:});
end

persistent INFO;
if isempty(INFO)
    INFO = get_info;
end
nnassert.minargs(nargin,1);
in1 = varargin{1};
if ischar(in1)
    switch (in1)
        case 'info'
            out1 = INFO;
        case 'apply'
            [out1,out2] = train_network(varargin{2:end});
        case 'formatNet'
            out1 = formatNet(varargin{2});
        case 'check_param'
            param = varargin{2};
            err = nntest.param(INFO.parameters,param);
            if isempty(err)
                err = check_param(param);
            end
            if nargout > 0
                out1 = err;
            elseif ~isempty(err)
                nnerr.throw('Type',err);
            end
        otherwise
            try
                out1 = eval(['INFO.' in1]);
            catch 
                nnerr.throw(['Unrecognized first argument: ''' in1 ''''])
            end
    end
else
    net = varargin{1};
    oldTrainFcn = net.trainFcn;
    oldTrainParam = net.trainParam;
    if ~strcmp(net.trainFcn,mfilename)
        net.trainFcn = mfilename;
        net.trainParam = INFO.defaultParam;
    end
    [out1,out2] = train(net,varargin{2:end});
    net.trainFcn = oldTrainFcn;
    net.trainParam = oldTrainParam;
end
end

%  BOILERPLATE_END
%% =======================================================

function info = get_info()
isSupervised = true;
usesGradient = false;
usesJacobian = true;
usesValidation = true;
supportsCalcModes = true;
showWindow = ~isdeployed; % showWindow must be false if network is deployed
info = nnfcnTraining(mfilename,'Levenberg-Marquardt',8.0,...
    isSupervised,usesGradient,usesJacobian,usesValidation,supportsCalcModes,...
    [ ...
    nnetParamInfo('showWindow','Show Training Window Feedback','nntype.bool_scalar',showWindow,...
    'Display training window during training.'), ...
    nnetParamInfo('showCommandLine','Show Command Line Feedback','nntype.bool_scalar',false,...
    'Generate command line output during training.'), ...
    nnetParamInfo('show','Command Line Frequency','nntype.strict_pos_int_inf_scalar',25,...
    'Frequency to update command line.'), ...
    ...
    nnetParamInfo('epochs','Maximum Epochs','nntype.pos_int_scalar',1000,...
    'Maximum number of training iterations before training is stopped.'), ...
    nnetParamInfo('time','Maximum Training Time','nntype.pos_inf_scalar',inf,...
    'Maximum time in seconds before training is stopped.'), ...
    ...
    nnetParamInfo('goal','Performance Goal','nntype.pos_scalar',0,...
    'Performance goal.'), ...
    nnetParamInfo('min_grad','Minimum Gradient','nntype.pos_scalar',1e-7,...
    'Minimum performance gradient before training is stopped.'), ...
    nnetParamInfo('max_fail','Maximum Validation Checks','nntype.pos_int_scalar',6,...
    'Maximum number of validation checks before training is stopped.'), ...
    ...
    nnetParamInfo('mu','Mu','nntype.pos_scalar',0.001,...
    'Mu.'), ...
    nnetParamInfo('mu_dec','Mu Decrease Ratio','nntype.real_0_to_1',0.1,...
    'Ratio to decrease mu.'), ...
    nnetParamInfo('mu_inc','Mu Increase Ratio','nntype.over1',10,...
    'Ratio to increase mu.'), ...
    nnetParamInfo('mu_max','Maximum mu','nntype.strict_pos_scalar',1e10,...
    'Maximum mu before training is stopped.'), ...
    ], ...
    [ ...
    nntraining.state_info('gradient','Gradient','continuous','log') ...
    nntraining.state_info('mu','Mu','continuous','log') ...
    nntraining.state_info('val_fail','Validation Checks','discrete','linear') ...
    ]);
end

function err = check_param(~)
err = '';
end

function net = formatNet(net)
if isempty(net.performFcn)
    warning(message('nnet:train:EmptyPerformanceFixed'));
    net.performFcn = 'mse';
    net.performParam = mse('defaultParam');
end
if isempty(nnstring.first_match(net.performFcn,{'sse','mse'}))
    warning(message('nnet:train:NonSqrErrorFixed'));
    net.performFcn = 'mse';
    net.performParam = mse('defaultParam');
end
end

function [archNet,tr] = train_network(archNet,rawData,calcLib,calcNet,tr)
[archNet,tr] = nnet.train.trainNetwork(archNet,rawData,calcLib,calcNet,tr,localfunctions);
end

function worker = initializeTraining(archNet,calcLib,calcNet,tr)

% Cross worker existence required
worker.WB2 = [];

% Initial Gradient
[worker.perf,worker.vperf,worker.tperf,worker.je,worker.jj,worker.gradient] = calcLib.perfsJEJJ(calcNet);

if calcLib.isMainWorker
    
    % Training control values
    worker.epoch = 0;
    worker.startTime = clock;
    worker.param = archNet.trainParam;
    worker.originalNet = calcNet;
    [worker.best,worker.val_fail] = nntraining.validation_start(calcNet,worker.perf,worker.vperf);
    
    worker.WB = calcLib.getwb(calcNet);
    worker.lengthWB = length(worker.WB);
    
    worker.ii = sparse(1:worker.lengthWB,1:worker.lengthWB,ones(1,worker.lengthWB));
    worker.mu = worker.param.mu;
    
    % Training Record
    worker.tr = nnet.trainingRecord.start(tr,worker.param.goal,...
        {'epoch','time','perf','vperf','tperf','mu','gradient','val_fail'});
    
    % Status
    worker.status = ...
        [ ...
        nntraining.status('Epoch','iterations','linear','discrete',0,worker.param.epochs,0), ...
        nntraining.status('Time','seconds','linear','discrete',0,worker.param.time,0), ...
        nntraining.status('Performance','','log','continuous',worker.perf,worker.param.goal,worker.perf) ...
        nntraining.status('Gradient','','log','continuous',worker.gradient,worker.param.min_grad,worker.gradient) ...
        nntraining.status('Mu','','log','continuous',worker.mu,worker.param.mu_max,worker.mu) ...
        nntraining.status('Validation Checks','','linear','discrete',0,worker.param.max_fail,0) ...
        ];
end
end

function [worker,calcNet] = updateTrainingState(worker,calcNet)

% Stopping Criteria
current_time = etime(clock,worker.startTime);
[userStop,userCancel] = nntraining.stop_or_cancel();
if userStop
    worker.tr.stop = message('nnet:trainingStop:UserStop');
    calcNet = worker.best.net;
elseif userCancel
    worker.tr.stop = message('nnet:trainingStop:UserCancel');
    calcNet = worker.originalNet;
elseif (worker.perf <= worker.param.goal)
    worker.tr.stop = message('nnet:trainingStop:PerformanceGoalMet');
    calcNet = worker.best.net;
elseif (worker.epoch == worker.param.epochs)
    worker.tr.stop = message('nnet:trainingStop:MaximumEpochReached');
    calcNet = worker.best.net;
elseif (current_time >= worker.param.time)
    worker.tr.stop = message('nnet:trainingStop:MaximumTimeElapsed');
    calcNet = worker.best.net;
elseif (worker.gradient <= worker.param.min_grad)
    worker.tr.stop = message('nnet:trainingStop:MinimumGradientReached');
    calcNet = worker.best.net;
elseif (worker.mu >= worker.param.mu_max)
    worker.tr.stop = message('nnet:trainingStop:MaximumMuReached');
    calcNet = worker.best.net;
elseif (worker.val_fail >= worker.param.max_fail)
    worker.tr.stop = message('nnet:trainingStop:ValidationStop');
    calcNet = worker.best.net;
end

% Training Record
worker.tr = nnet.trainingRecord.update(worker.tr, ...
    [worker.epoch current_time worker.perf worker.vperf worker.tperf worker.mu worker.gradient worker.val_fail]);
worker.statusValues = ...
    [worker.epoch,current_time,worker.best.perf,worker.gradient,worker.mu,worker.val_fail];
end

function [worker,calcNet] = trainingIteration(worker,calcLib,calcNet)

% Cross worker control variables
muBreak = [];
perfBreak = [];

% Levenberg Marquardt
while true
    
    %SA
    try
        work = evalin('base','workerRecord');
    catch
        work = {};
    end    
    fprintf('epoch %d: worker.perf %f vperf %f tperf %f\n',...
        worker.epoch,worker.perf,worker.vperf,worker.tperf);
    
    
    
    if calcLib.isMainWorker
        muBreak = (worker.mu > worker.param.mu_max);
    end
    if calcLib.broadcast(muBreak)
        break
    end
    
    if calcLib.isMainWorker
        % Check for Singular Matrix
        [msgstr,msgid] = lastwarn;
        lastwarn('MATLAB:nothing','MATLAB:nothing')
        warnstate = warning('off','all');
        dWB = -(worker.jj + worker.ii * worker.mu) \ worker.je;
        [~,msgid1] = lastwarn;
        flag_inv = isequal(msgid1,'MATLAB:nothing');
        if flag_inv
            lastwarn(msgstr,msgid);
        end
        warning(warnstate)
        worker.WB2 = worker.WB + dWB;        
    end
    
    
    %SA
    work{end+1} = worker;    
    assignin('base','workerRecord',work);
        
    
    calcNet2 = calcLib.setwb(calcNet,worker.WB2);
    perf2 = calcLib.trainPerf(calcNet2);
    
    if calcLib.isMainWorker
        perfBreak = (perf2 < worker.perf) && flag_inv;
    end
    if calcLib.broadcast(perfBreak)
        worker.WB = worker.WB2;
        calcNet = calcNet2;
        if calcLib.isMainWorker
            worker.mu = max(worker.mu * worker.param.mu_dec,1e-20);
        end
        break
    end
    
    if calcLib.isMainWorker
        worker.mu = worker.mu * worker.param.mu_inc;
    end

end

% Track Best Network
[worker.perf,worker.vperf,worker.tperf,worker.je,worker.jj,worker.gradient] = calcLib.perfsJEJJ(calcNet);
if calcLib.isMainWorker
    [worker.best,worker.tr,worker.val_fail] = nnet.train.trackBestNetwork(...
        worker.best,worker.tr,worker.val_fail,calcNet,worker.perf,worker.vperf,worker.epoch);
end
end