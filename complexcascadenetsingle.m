classdef complexcascadenetsingle < handle
    % complexnet Feed-forward neural network with variety of
    % backpropagation approaches
    %
    % Swaroop Appadwedula Gr.64     % August 13,2020 in the time of COVID
    
    properties
        hiddenSize   % vector of hidden layer sizes, not including output
                     % empty when single layer                   
        layers       % cell array containing transferFcn spec, etc
                     % layers{ll}.transferFcn
        nbrofLayers  % number of layers = hidden + 1 
        nbrofUnits   % 2xnbrofLayers matrix 
                     % with row1=input dim, row2=output dim
                     
        % count of #of weights and # bias in each layer                     
        nbrofLayerWeights
        nbrofInputWeights         
        nbrofBias        
        nbrofWeights
        nbrofParameters
                
        inputConnections %'full','firstlast'
        layerConnections %'full','next'
        
        % [layers x 1] binary connections between bias and layer
        biasConnect              
        % [layers x 1] binary connections between input and layers (first element=1)
        inputConnect      
        % [layers x layers] binary connections between layers
        layerConnect              
        
        inputweightinds   % how weights and bias can be vectorized
        layerweightinds  
        layerbiasinds                
        
        InputWeights
        LayerWeights
        bias
        
        lrate        
        
        % keep a history of weights as a vector
        WeightsBuffer  % buffer of last max_fail weights

        trainFcn     % see options in gradient_descent directory
                     % Hessian-based 'trainlm'=Levenberg-Marquardt
                     % with Bayesian regularization 'trainbr'

              
        % Levenberg-Marquardt parameter for diagonal load to mix Hessian
        % step with gradient step
        mu
        mu_inc       % increment diagonal load when mse increases
        mu_dec
        mu_max       % bound the values of the diagonal load
        mu_min

        % Bayesian-Regularization parameters updated over time
        beta         % estimate of 1/variance{data} 
        alpha        % estimate of 1/variance{weights}
        gamma        % number of good parameter measurements
                     % between (0, number of weights + biases )        
                 
        initFcn      % 'randn','nguyen-widrow'

        minbatchsize
        batchtype    %'randperm','fixed','index'
         
        % for batchtype 'index' where indices are provided in params
        trainInd
        valInd
        testInd
        nbrofEpochs
        
        % mapminmax
        domap           %'reim','complex' or 0
        inputSettings
        outputSettings        
        dorealifyfn     % either y=x or break into re/im for mapminmax
        dounrealifyfn
        
        % for debugging
        debugPlots
        printmseinEpochs  % if 1, print every time
        performancePlots
                
        % for comparison with complexnet
        debugGradient
        someRecord        
    end
    
    properties (Constant)
        nbrofEpochsdefault = 1e3; % number of iterations picking a batch each time and running gradient        
        lrate_default = 1e-2; % initial step size for the gradient        

        lm_mu = 0.001; % Hessian + mu * eye for Levenberg-Marquardt
        br_mu = 0.001; % Bayesian regularization
               
        batchsize_per_feature = 50;    % number of samples per feature to use at a time in a epoch
        minbatchsizedefault = 25;
        batchtypedefault = 'randperm';
        
        epochs_drop = 100;   % number of epochs before dropping learning rate
        drop = 1/10;         % drop in learning rate new = old*drop
        
        domapdefault = 'complex'  % mapminmax( complex ) instead of re/im
        
        % some stopping parameters
        %maxtime = 60*200;    % (s) compared to tic/toc
        maxtime = Inf;        % (s) let epochs determine stopping        
        
        msedesired = 0;
        min_grad = 1e-7 / 10; % matlab sets to 1e-7, set lower due to complex
                             % @todo: unclear where this number comes from
        max_fail = 100;        % allow max_fail steps of increase in 
                             % validation data before stopping 
    end        
        
    methods (Static)    
                
        function y=realify(x)
            y=[real(x); imag(x)];
        end
        function x=unrealify(y)
            if size(y,1)/2 ~= floor(size(y,1)/2)
                error('unrealify takes first half of features + 1i * second half of features');
            else
                x=y(1:end/2,:) + 1i*y(end/2+1:end,:);
            end
        end
        
        
        % complex randn for intializing weights
        function z=crandn(m,n)
            z=complex(randn(m,n),randn(m,n));
        end        
        function z=crand(m,n)
            z=complex(2*rand(m,n)-1,2*rand(m,n)-1);
        end        
        
        function [z,dz] = square(x)
            z = x.*x;
            dz = 2 * x;
        end
        
        % out = in linear, keeps real and imag separate
        function [z,dz]=purelin(x)
            z=x;
            dz=ones(size(x));
            %dz.real=ones(size(x));
            %dz.imag=ones(size(x));            
        end
        
        % phase only
        function [z,dz]=purephase(x)
            z=x./abs(x);
            % function not differentiable, so use equal effect from all x
            % http://makeyourownneuralnetwork.blogspot.com/2016/05/complex-valued-neural-networks.html
            dz=ones(size(x));
        end

        % saturate linear
        % clip real and imag at +/-a
        function [z,dz]=satlins(x)            
            a = 1;
            zr = real(x); zi = imag(x);                        
            indsr = abs(zr)> a;
            indsi = abs(zi)> a;
            zr(indsr) = a * sign(zr(indsr));
            zi(indsi) = a * sign(zi(indsi));            
            z = zr + 1i * zi;
            
            % assume that derivative is 1 when either is linear
            [dz.real,dz.imag] = deal(zeros(size(z)));
            dz.real(~indsr) = 1;
            dz.imag(~indsi) = 1;                        
        end

        % complex ReLU
        % CReLU (Trabelsi et al., 2017) ReLU(Re z) + i ReLU(Im z)
        function [z,dz]=cartrelu(x)            
            zr = real(x); zi = imag(x);                        
            indsr = zr< 0;
            indsi = zi< 0;
            zr(indsr) = 0;
            zi(indsi) = 0;
            z = zr + 1i * zi;
            
            % assume that derivative is 1 when either is linear
            % @todo: should this be treated as split re/im or not            
            
            % treating as split re/im
            [dz.real,dz.imag] = deal(zeros(size(z)));
            dz.real(~indsr) = 1;
            dz.imag(~indsi) = 1;  
            
            % treating as a complex function does not work!
            %dz = zeros(size(z));
            %dz(~indsr) = 1;            
            %dz(~indsi) = dz(~indsi) + 1i;            
        end        

        % two sided sigmoidal        
        function [z,dz]=mytansig(x)           
            if 0
                cval = 2; kval = 2; lval = -1;
                z =  cval./(1+exp(-kval*x));
                dz = kval * (z.*(1-z/cval));
                z = z  + lval;
            elseif 0        
                % tansig is the same as cval=2, kval=2, lval = -1                
                lval = -1;
                z = tansig(x) - lval;      % take out lval for easy dz calculation
                dz = 2 *( z.*(1 - z / 2) );
                z = z  + lval;             % put back in lval
            else
                z = tansig(x);
                dz = tansig('da_dn',x);
            end            
            %{ 
            %check with matlab
            x = -10:0.001:10;
            cval = 2; kval = 2; lval = -1;
            z =  cval./(1+exp(-kval*x));
            dz = kval * (z.*(1-z/cval));                        
            z = z  + lval;
            figure(1);
            zz=tansig(x); zzp=diff(zz)./diff(x); zzp=tansig('da_dn',x);
            subplot(211);plot(x,z,'.'); hold on; plot(x,zz,'o');
            subplot(212);plot(x,dz,'.'); hold on; plot(x(1:end-1),zzp(1:end-1),'o');
            %}
        end
        
        % single sided sigmoidal
        function [z,dz]=sigrealimag(x)
            zr = ( 1./(1+exp(-real(x))) );
            zi = ( 1./(1+exp(-imag(x))) );
            z =  zr + 1i*zi;
            % derivative is exp(-x)./(1+ exp(-x)).^2
            % which can be written as z.*(1-z);
            dz.real = zr.*(1-zr);
            dz.imag = zi.*(1-zi);
        end
        
        % two sided sigmoidal
        % cval,kval suggested by Benvenuto and Piazza 
        % "On the complex backprop alg" Trans Signal Processing
        function [z,dz]=sigrealimag2(x)
            if 1
                cval = 2; kval = 2; lval = -1;
                zr = ( cval./(1+exp(-kval*real(x))) );
                zi = ( cval./(1+exp(-kval*imag(x))) );
                z = (zr + lval) + 1i*(zi + lval);
                dz.real = kval * zr.*(1-zr/cval);
                dz.imag = kval * zi.*(1-zi/cval);
            elseif 0
                % tansig is the same as cval=2, kval=2, lval = -1
                lval = -1;
                zr = tansig(real(x)) - lval;      % take out lval for easy dz calculation
                zi = tansig(imag(x)) - lval;      % take out lval for easy dz calculation
                dz.real = 2 *( zr.*(1 - zr / 2) );
                dz.imag = 2 *( zi.*(1 - zi / 2) );
                zr = zr  + lval;             % put back in lval
                zi = zi  + lval;             % put back in lval
                z = (zr + lval) + 1i*(zi + lval);
            else
                z = tansig(real(x)) +1i * tansig(imag(x));
                dz.real = tansig('da_dn',real(x));
                dz.imag = tansig('da_dn',imag(x));
            end
        end
        
        % tanh
        % use tansig instead since mathematically equivalent
        function [z,dz]=mytanh(x)           
            z = tanh(x);
            dz = 1 - z.^2;
        end
        
        % asinh 
        function [z,dz]=myasinh(x)           
            z = asinh(x);
            dz = 1./sqrt(1 + x.^2);
        end
        
        % split real/imag tanh
        function [z,dz]=carttanh(x)  
            zr = tanh(real(x));
            zi = tanh(imag(x));
            z =  zr + 1i * zi;
            dz.real = 1 - zr.^2; dz.imag = 1-zi.^2;
        end        
        
    end
    methods        
        function obj = complexcascadenetsingle(params)
            % complexnet Construct an instance of this class
            % assign the transfer function
            
            if ~exist('params','var')
                params = struct();
            end
            if ~isfield(params,'hiddenSize'), params.hiddenSize = []; end
            if ~isfield(params,'outputFcn'), params.outputFcn = 'purelin'; end
            if ~isfield(params,'layersFcn'), params.layersFcn = 'sigrealimag2'; end            
            if isfield(params,'minbatchsize')
                obj.minbatchsize = params.minbatchsize; 
            else
                obj.minbatchsize = obj.minbatchsizedefault;
            end
                        
            if isfield(params,'batchtype')
                obj.batchtype= params.batchtype; 
            else
                obj.batchtype = obj.batchtypedefault;
            end
            
            switch obj.batchtype
                case 'index'
                    obj.trainInd = params.trainInd;
                    obj.valInd = params.valInd;
                    obj.testInd = params.testInd;            
            end
            
            if isfield(params,'lrate')
                obj.lrate= params.lrate; 
            else
                obj.lrate=obj.lrate_default;
            end
            
            if isfield(params,'mu')
                obj.mu = params.mu; 
            else
                obj.mu = obj.lm_mu;                
            end
            
            if isfield(params,'mu_inc')
                obj.mu_inc= params.mu_inc; 
            else
                obj.mu_inc = 10;                
            end
            if isfield(params,'mu_dec')
                obj.mu_dec= params.mu_dec; 
            else
                obj.mu_dec = 1/10;                
            end
            if isfield(params,'mu_max')
                obj.mu_max= params.mu_max; 
                obj.mu_min= params.mu_min;                 
            else
                obj.mu_max = 1e10;      
                obj.mu_min = 1e-20;
            end
            
            if obj.mu_inc<1, error('mu_inc must be >=1'); end
            if obj.mu_dec>1, error('mu_dec must be <=1'); end            
            
            if isfield(params,'nbrofEpochs')
                obj.nbrofEpochs= params.nbrofEpochs;
            else
                obj.nbrofEpochs = obj.nbrofEpochsdefault;
            end
            
            % type of training
            if ~isfield(params,'initFcn'), params.initFcn='nguyen-widrow'; end
            obj.initFcn = params.initFcn;
            if ~isfield(params,'trainFcn'), params.trainFcn='Adadelta'; end
            obj.trainFcn = params.trainFcn;
            
            % map to [-1 1] if specifically requested
            % otherwise, no mapping if linear
            %@todo: this should be done based on ranges of the functions
            if iscell(params.layersFcn)
                lFcn = params.layersFcn{1};
            else
                lFcn = params.layersFcn;
            end
            pp = strcmp(lFcn,'purelin');
            if isfield(params,'domap') 
                obj.domap=params.domap; 
                if pp==1 && ~obj.domap==0, warning('mapping to [-1 1] used even though layers are linear'); end
            elseif pp==1
                % no mapping if linear
                obj.domap = 0;
            else
                % use mapping if nonlinear
                obj.domap = 1;
            end            
            switch obj.domap
                case 1, obj.domap = obj.domapdefault;
            end
                                
            if ~isfield(params,'biasConnect'),params.biasConnect=[]; end                        
            if ~isfield(params,'layerConnect'),params.layerConnect=[]; end            
            if ~isfield(params,'inputConnect'),params.inputConnect=[]; end                        
            obj.biasConnect = params.biasConnect;
            obj.layerConnect = params.layerConnect;
            obj.inputConnect = params.inputConnect;                        
            
            if isfield(params,'debugPlots')
                obj.debugPlots=params.debugPlots;
            else
                obj.debugPlots=0;
            end
            if isfield(params,'performancePlots')
                obj.performancePlots=params.performancePlots;
            else
                obj.performancePlots=1;
            end
            
            
            
            if isfield(params,'printmseinEpochs')
                obj.printmseinEpochs=params.printmseinEpochs;
            else
                obj.printmseinEpochs=10;
            end                                        
            % hidden layers sizes is a vector 
            obj.hiddenSize = params.hiddenSize;
            obj.layers = cell(1,length(obj.hiddenSize)+1);

            outputFcn = params.outputFcn;
            layersFcn = params.layersFcn;
            
            % hidden layers
            for ll=1:length(obj.layers)-1
                if iscell(layersFcn)
                    if numel(layersFcn)==length(obj.layers)-1
                        obj.layers{ll}.transferFcn = layersFcn{ll};
                    end
                elseif char(layersFcn)
                    obj.layers{ll}.transferFcn = layersFcn;
                else
                    error('Unknown layersFcn variable type');
                end
                % each weight matrix is output size x input size since it
                % is applied on the left of the input vector
                % obj.Weights{ll} * x{ll}                
            end
            % output layer
            obj.layers{end}.transferFcn = outputFcn;            
            obj.nbrofLayers = numel(obj.hiddenSize) + 1;            
            %[obj.SumDeltaWeights, obj.SumDeltaWeights2] = deal(cell(1,obj.nbrofLayers));            
                        
            switch obj.initFcn
                case 'previous'
                    % use existing weights assigned previously to this call
                otherwise
                    obj.bias = cell(obj.nbrofLayers,1);                        
                    obj.InputWeights = cell(obj.nbrofLayers,1);
                    obj.LayerWeights = cell(obj.nbrofLayers,obj.nbrofLayers);
            end
            
        end

        %----------------------------------------------------------
        % evaluate the network
        % given an input, inputaffine mapping (via mapminmax), and layers
        % (non)linear functions
        % determine the network output, intermediate terms, and derivative
        %
        % in   matrix
        % out  matrix with same second dimension (samples) as in
        % n    cell array of nbrofLayers with network values (matrices)
        %      sensitivity is computed as derror^2/dn
        % a0   matrix is mapminmax(in)
        % a    cell array of nbrofLayers with input matrices at each layer
        % fdot cell array of derivatives of the activation function at f(n)
        function [out,n,a0,a,fdot] = test(obj,in)
            
            [n,a,fdot] = deal(cell(obj.nbrofLayers,1));
            out = []; a0 = [];            
            
            if isempty(in), return; end
            
            switch obj.domap
                case 0
                    % gain only
                    %in = bsxfun(@times,in,obj.inputSettings.gain);
                    % no mapping
                    a0 = in;
                otherwise
                    % scale the input
                    ins = mapminmax.apply(obj.dorealifyfn(in),obj.inputSettings);
                    a0 = obj.dounrealifyfn(ins);            
            end
            
            for layer=1:obj.nbrofLayers                
                % transfer function
                transferFcn = obj.layers{layer}.transferFcn;  % string
                activationfn = @(x) obj.(transferFcn)(x);  % handle

                % weight matrix from input to current layer
                W=obj.InputWeights{layer};                       
                if ~isempty(W)
                    %---- weight ----
                    n{layer} = W*a0;
                    %----------------
                else
                    n{layer} = [];
                end
                            
                % from previous layers (nn,nn-1) is the usual connection,
                % the others are due to cascade architecture
                for fromlayer=1:layer-1
                    % weight matrix for previous layers
                    W=obj.LayerWeights{layer,fromlayer};
                    if ~isempty(W) 
                        if isempty(a{fromlayer})
                            error('test: a{%d} empty for layer weight{%d,%d}\n',...
                                fromlayer,layer,fromlayer)
                        elseif isempty(n{layer})
                            %---- weight ----
                            n{layer} = W*a{fromlayer};
                            %----------------
                        else
                            %---- weight ----
                            toadd = W*a{fromlayer};
                            %----------------
                            if any( size(toadd) ~= size(n{layer}))
                            error('test: n{%d} from InputWeights{%d} does not match LayerWeights{%d,%d}\n',...
                                layer,layer,layer,fromlayer);
                            else
                                n{layer} = n{layer} + toadd;
                            end
                        end
                    end
                end
                
                % bias at each layer is needed to move input into different
                % parts of the nonlinearity
                b=obj.bias{layer};
                if ~isempty(b)
                    if ~isempty(n{layer})                        
                        %---- bias ----
                        n{layer} = n{layer} + b;
                        %----------------
                    else
                        error('test: n{%d} empty but bias{%d} connected\n',layer,layer);
                    end
                end
                
                % evaluate f(xn) and f'(xn)
                [a{layer},fdot{layer}] = activationfn(n{layer});                
            end
               
            out_normalized = a{obj.nbrofLayers};
            switch obj.domap
                case 0
                    % gain only
                    %out = bsxfun(@rdivide,out_normalized,obj.outputSettings.gain);
                    %no scaling
                    out=out_normalized;
                otherwise
                    % network matches to -1,1 but output is back to original scale
                    out = mapminmax.reverse( obj.dorealifyfn(out_normalized),obj.outputSettings);
                    out = obj.dounrealifyfn(out);
            end
        end
                
        %------------------------------------------------------------------
        % copy weights from complexnet and initialize connections to make
        % casacade into plain feedforward
        % set initFcn to previous to allow for starting weights to be setup
        % the same was as complexnet        
        function objcascade = copycomplexnet(objcascade,objcomplex,params)
            nbrofLayers = objcomplex.nbrofLayers;
            params.inputConnect = zeros(nbrofLayers,1); params.inputConnect(1) = 1;
            params.layerConnect = zeros(nbrofLayers,nbrofLayers);
            for tolayer=2:nbrofLayers, params.layerConnect(tolayer,tolayer-1) = 1; end            
            params.biasConnect = ones(nbrofLayers,1);
            
            objcascade.someRecord = objcomplex.someRecord;
            
            objcascade.bias = cell(1,nbrofLayers);
            for layer = 1:nbrofLayers
                objcascade.bias{layer} = objcomplex.Weights{layer}(:,end);
                if layer==1
                    objcascade.InputWeights{1} = objcomplex.Weights{1}(:,1:end-1);
                else
                    objcascade.LayerWeights{layer,layer-1} = objcomplex.Weights{layer}(:,1:end-1);
                end
                objcascade.layers{layer} = objcomplex.layers{layer};
            end
            objcascade.initFcn = 'previous';
            fprintf('Initialized obj.initFcn to %s for copied weights to be used as initial weights\n',objcascade.initFcn);
        end
        
        %------------------------------------------------------------------
        % convert WbWb vectors used in complexnet to bIWLW vectors 
        % used in complexcascadenet assuming feedforward architecture        
        function bIWLW = WbWb_to_bIWLW(obj,WbWb)            
            bIWLW = zeros(size(WbWb));
            inds = obj.inputweightinds{1};
            num = length(inds); ii = 1:num;
            bIWLW(inds,:) = WbWb(ii,:);

            inds = obj.layerbiasinds{1};
            num = length(inds); ii = ii(end) + (1:num);            
            bIWLW(inds,:) = WbWb(ii,:);
            
            for tolayer = 2:obj.nbrofLayers                   
                fromlayer = tolayer-1;              
                inds = obj.layerweightinds{tolayer,fromlayer};
                num = length(inds); ii = ii(end) + (1:num);
                bIWLW(inds, :) = WbWb(ii,:);                      

                inds = obj.layerbiasinds{tolayer};
                num = length(inds); ii = ii(end) + (1:num);                
                bIWLW(inds,:) = WbWb(ii,:);
            end            
        end
        
        %------------------------------------------------------------------
        function obj = copynet(obj,net,workerRecord,IW,LW,b)
            % copy over matlab weights            

            if isempty(obj.Weights)
                obj.Weights = cell(1,obj.nbrofLayers);
            end
            
            if exist('IW','var') && exist('LW','var') && exist('b','var')
                fprintf('Using input IW,LW,b\n');
                %already assigned
            elseif exist('workerRecord','var')
                fprintf('Using IW,LW,b from workerRecord{1}.WB\n');                
                % these are the weights in the weight record
                [b,IW,LW] = separatewb(net,workerRecord{1}.WB);                
            else
                warning('No workerRecord or IW,LW,b.  Using FINAL weights');
                IW = net.IW;  % these are the final values of weights
                LW = net.LW;
                b = net.b;
            end
              
            if exist('workerRecord','var')
                obj.workerRecord = workerRecord;
            end
            
            obj.InputWeights = IW;
            obj.LayerWeights = LW;
            obj.bias = b;
            obj.initFcn = 'previous';
            fprintf('Initialized obj.initFcn to %s for copied weights to be used as initial weights\n',obj.initFcn);
        end


        %------------------------------------------------------------------        
        % Jacobian contain derivatives of all parameters vectorized instead
        % of each set of weights in a cell
        % follow matlab convention to have bias first, then weights from
        % input then weights from previous layer
        function [obj] = getlayerinds(obj)
            
            % organize the terms in the Jacobian
            [obj.inputweightinds,obj.layerbiasinds] = deal(cell(obj.nbrofLayers,1));
            obj.layerweightinds = deal(cell(obj.nbrofLayers,obj.nbrofLayers));
            
            currentindex = 1;
            for layer = 1:obj.nbrofLayers                
                % ordering is stacking one layer at a time from input
                
                if obj.biasConnect(layer)
                % bias comes first
                L = numel(obj.bias{layer});
                obj.layerbiasinds{layer} = currentindex-1 + (1:L);
                currentindex = currentindex + L;
                end
                
                if obj.inputConnect(layer)
                % input weights going into each layer
                L = numel(obj.InputWeights{layer});
                obj.inputweightinds{layer} = currentindex-1 + (1:L);
                currentindex = currentindex + L;
                end
                
                % layer weights, including weights directly connecting
                % previous layers
                for fromlayer = 1:layer-1
                    if obj.layerConnect(layer,fromlayer)
                    L=numel(obj.LayerWeights{layer,fromlayer});
                    obj.layerweightinds{layer,fromlayer} = currentindex-1 + (1:L);
                    currentindex = currentindex + L;
                    end
                end
            end 
        end        
        
        
        % input is InputWeights{layer}, LayerWeights{tolayer,fromlayer}, 
        % bias{layer} 
        % output is [b IW LW] stacked over the layers
        function bIWLW = Weights_to_vec(obj,bias,InputWeights,LayerWeights)
            nbrofSamples = size(bias{1},3);
            ii = 1:nbrofSamples;
            bIWLW = zeros(sum(obj.nbrofBias+obj.nbrofWeights),nbrofSamples);
            for tolayer = 1:obj.nbrofLayers
                if obj.biasConnect(tolayer)
                    bIWLW( obj.layerbiasinds{tolayer},ii) = ...
                        reshape( bias{tolayer},[],nbrofSamples);
                end
                if obj.inputConnect(tolayer)
                    bIWLW( obj.inputweightinds{tolayer}, ii) = ...
                        reshape( InputWeights{tolayer},[],nbrofSamples);
                end
            
                for fromlayer = 1:tolayer-1
                   if obj.layerConnect(tolayer,fromlayer)
                    bIWLW( obj.layerweightinds{tolayer,fromlayer}, ii) = ...
                        reshape( LayerWeights{tolayer,fromlayer},[],nbrofSamples);
                   end
                end
            end            
        end
        
        % input is vectorized [b IW LW] stacked over the layers
        % output is InputWeights{layer}, LayerWeights{tolayer,fromlayer}, 
        % bias{layer} 
        
        function [bias,InputWeights,LayerWeights] = vec_to_Weights(obj,bIWLW)
            nbrofSamples = size(bIWLW,2);
            ii = 1:nbrofSamples;
            [InputWeights, bias] = deal(cell(obj.nbrofLayers,1));
            LayerWeights = cell(obj.nbrofLayers,obj.nbrofLayers);
            
            for tolayer=1:obj.nbrofLayers
                nbrofNeurons = obj.nbrofUnits(2,tolayer);
                nbrofIn = obj.nbrofUnits(1,1);
                if obj.biasConnect(tolayer)
                    bias{tolayer} = ...
                        reshape( bIWLW( obj.layerbiasinds{tolayer}, ii), nbrofNeurons,nbrofSamples);
                end
                if obj.inputConnect(tolayer)
                    InputWeights{tolayer} = ...
                        reshape(bIWLW( obj.inputweightinds{tolayer}, ii), nbrofNeurons,nbrofIn,nbrofSamples);
                end                
                for fromlayer = 1:tolayer-1
                    if obj.layerConnect(tolayer,fromlayer)
                        nbrofNeurons = obj.nbrofUnits(2,tolayer);
                        nbrofIn = obj.nbrofUnits(2,fromlayer);
                        LayerWeights{tolayer,fromlayer} = ...
                            reshape(bIWLW( obj.layerweightinds{tolayer,fromlayer}, ii),nbrofNeurons,nbrofIn,nbrofSamples);
                    end
                end
            end            
        end                
        
        function obj = updateWeights(obj, Deltab, DeltaIW, DeltaLW, scale)
            for layer=1:obj.nbrofLayers
                if obj.biasConnect(layer)
                obj.bias{layer} = obj.bias{layer} +  scale * Deltab{layer};
                end
                if obj.inputConnect(layer)
                obj.InputWeights{layer} = obj.InputWeights{layer} +  scale * DeltaIW{layer};
                end                
                for fromlayer = 1:layer-1
                    if obj.layerConnect(layer,fromlayer)
                    obj.LayerWeights{layer,fromlayer} = ...
                        obj.LayerWeights{layer,fromlayer} + scale * DeltaLW{layer,fromlayer};
                    end
                end                
            end
        end
        
        % initialize the weights and the bias
        % many optimization approaches are sensitive 
        % to the choice of initial weights
        % multiple trials are usually best to get best performance
        function [w,b] = init(obj,nbrofNeurons,nbrofIn)            
            switch obj.initFcn
                %-----------------------------
                % complex initializations
                case 'crandn'
                    w = 0.1*crandn( nbrofNeurons, nbrofIn );
                    b = 0.1*crandn( nbrofNeurons, 1);
                case 'crands'
                    w = rands( nbrofNeurons, nbrofIn ) + 1i*rands( nbrofNeurons, nbrofIn );      
                    b = rands( nbrofNeurons, 1) + 1i*rands( nbrofNeurons, 1); 
                case 'c-nguyen-widrow'
                    % borrowed Matlab's implementation of Nguyen-Widrow
                    w = zeros(  nbrofNeurons, nbrofIn );
                    activeregionofactivation = [-2 2];  % tansig('active');
                    mapregion = [-1 1];
                    mapregions = repmat(mapregion,nbrofIn,1);
                    [wr,br]=mycalcnw(mapregions,nbrofNeurons,activeregionofactivation);
                    [wi,bi]=mycalcnw(mapregions,nbrofNeurons,activeregionofactivation);
                    w = wr+1i*wi;
                    b = br+1i*bi;
                    
                    %-----------------------------
                    % non-complex initializations
                case 'rands'
                    w = rands( nbrofNeurons, nbrofIn );         
                    b = rands( nbrofNeurons, 1);
                case 'randn'
                    w = 0.1*randn( nbrofNeurons, nbrofIn );         
                    b = 0.1*randn( nbrofNeurons, 1);
                case 'nguyen-widrow'
                    % borrowed Matlab's implementation of Nguyen-Widrow
                    w = zeros(  nbrofNeurons, nbrofIn );
                    activeregionofactivation = [-2 2];  % tansig('active');
                    mapregion = [-1 1];
                    mapregions = repmat(mapregion,nbrofIn,1);
                    [w,b]=mycalcnw(mapregions,nbrofNeurons,activeregionofactivation);                    
              
                otherwise
                    error('unknown initFcn %s for weights',obj.initFcn);
            end
        end
        
        
        
        %------------------------------------------------------------------
        function obj = train(obj,in,out)
            % in  is features x number of training samples
            % out is desired output x number of training samples            
                        
            switch obj.domap
                case 'reim'
                    obj.dorealifyfn = @(x) obj.realify(x);
                    obj.dounrealifyfn = @(x) obj.unrealify(x);
                case {'complex',0}
                    obj.dorealifyfn = @(x) x;
                    obj.dounrealifyfn = @(x) x;
            end            
            
            switch obj.domap
                case 0
                    % use gain only map
                    %obj.inputSettings.gain = 1./max(abs(in),[],2);
                    %obj.outputSettings.gain = 1./max(abs(out),[],2);
                    %in_normalized = bsxfun(@times,in,obj.inputSettings.gain);
                    %out_normalized = bsxfun(@times,out,obj.outputSettings.gain);
                    
                    % no normalization for real or imag
                    obj.inputSettings.gain = ones(size(in,1),1);
                    obj.outputSettings.gain = ones(size(out,1),1);
                    in_normalized=in;
                    out_normalized=out;                    
                    derivative_outputmap = diag(1./obj.outputSettings.gain);
                otherwise
                    % per feature normalization into -1 1
                    % y = (ymax-ymin)*(x-xmin)/(xmax-xmin) + ymin
                    % y = ymin + gain * (x - xmin)
                    % dx_dy = 1/gain
                    [in_normalized,obj.inputSettings] = mapminmax(obj.dorealifyfn(in));
                    [out_normalized,obj.outputSettings] = mapminmax(obj.dorealifyfn(out));
                    in_normalized = obj.dounrealifyfn(in_normalized);
                    out_normalized = obj.dounrealifyfn(out_normalized);
                    
                    derivative_outputmap = mapminmax('dx_dy',obj.dorealifyfn(out(:,1)),obj.dorealifyfn(out_normalized(:,1)),obj.outputSettings);
                    derivative_outputmap = cell2mat(derivative_outputmap);
            end
            
            [nbrofInUnits, nbrofSamples] = size(in);
            [nbrofOutUnits, nbr] = size(out);
            if nbr~=nbrofSamples
                error('input and output number of samples must be identical');
            end            
            nbrOfNeuronsInEachHiddenLayer = obj.hiddenSize;            
            if floor(obj.hiddenSize)~=obj.hiddenSize
                error('hiddenSize is not an integer');
            end            
            if nbrofOutUnits>1 && any(validatestring(obj.trainFcn,{'trainlm','trainbr'}))            
                error('complexcascadenet: multiple outputs not supported');
            end            
            
            % allocate space for gradients
            % sensitivity{n} = dcost/da{n} vector of dimension a{n}
            % sensitivity is propagated backwards using chain rule
            %
            % DeltaLW{n} = dcost/dLayerWeights{n} matrix of dimension LW{n}
            % DeltaIW{n} = dcost/dInputWeights{n} matrix of dimension IW{n}
            %
            % state is cell array with state of gradient and hessian approx
            % there is redundancy here since each layer has it's own state
            % that has rate parameters that are same - not a big deal
            [sensitivity,stateIW,DeltaIW,stateb,Deltab] = deal(cell(obj.nbrofLayers,1));
            [stateLW,DeltaLW] = deal(cell(obj.nbrofLayers,obj.nbrofLayers));            
                                   
            switch obj.trainFcn
                case {'trainlm','trainbr'}
                    obj.alpha = 0; 
                    % Hessian*beta, jace*beta matches matlab for trainlm
                    obj.beta = 1/2;  
            end
            switch obj.trainFcn
                case 'trainlm',obj.mu = obj.lm_mu;
                case 'trainbr',obj.mu = obj.br_mu;
            end
            
             switch obj.batchtype
                case {'randperm','fixed'}
                    switch obj.minbatchsize
                        case 'full'
                            % all samples used for training
                            nbrofSamplesinBatch = nbrofSamples;
                            nbrofSamplesinTest = 0;
                        case 'split70_15_15'
                            % 70% of samples used for training
                            % consistent with matlab
                            nbrofSamplesinBatch =  floor( 0.7 * nbrofSamples);
                            nbrofSamplesinTest = floor( 0.15 * nbrofSamples);
                        case 'split90_10'
                            nbrofSamplesinBatch =  floor( 0.9 * nbrofSamples);
                            nbrofSamplesinTest = 0;
                        case 'split70_30'
                            nbrofSamplesinBatch =  floor( 0.7 * nbrofSamples);
                            nbrofSamplesinTest = 0;
                        otherwise
                            % pick batch size number of sample if possible
                            nbrofSamplesinBatch =  max(obj.batchsize_per_feature*nbrofInUnits,obj.minbatchsize);
                            nbrofSamplesinBatch =  min(nbrofSamplesinBatch,nbrofSamples);
                            % testing and validation each get 1/2 of the remainder
                            % when remainder = 0 (e.g. params.minbatchsize =inf), then there is
                            % no test or validate data
                            nbrofSamplesinTest = floor( 1/2 * (nbrofSamples - nbrofSamplesinBatch) );
                    end
                    nbrofSamplesinValidate = nbrofSamples - nbrofSamplesinBatch - nbrofSamplesinTest;
                    
                case 'index'
                    nbrofSamplesinBatch = length(obj.trainInd);
                    nbrofSamplesinTest = length(obj.testInd);
                    nbrofSamplesinValidate = length(obj.valInd);                    
            end
          
            % -------------------------------------------------------------
            % the input (row1) and output (row2) dimensions at each layer
            % weights{layer} has dimension output x input 
            obj.nbrofUnits(1,:) = [nbrofInUnits  nbrOfNeuronsInEachHiddenLayer];  % input size
            obj.nbrofUnits(2,:) = [nbrOfNeuronsInEachHiddenLayer nbrofOutUnits];  % output size            
            obj.nbrofBias = obj.nbrofUnits(2,:); % bias                         
                        
            % default layer connect is layer->layer+1
            if isempty(obj.layerConnect) || ...
                    any( size(obj.layerConnect) ~= [obj.nbrofLayers,obj.nbrofLayers])                
                obj.layerConnections = 'all'; %'next'
                obj.layerConnect = zeros(obj.nbrofLayers,obj.nbrofLayers);
                for tolayer = 2:obj.nbrofLayers
                    switch obj.layerConnections
                        case 'all'                            
                            for fromlayer = 1:tolayer-1
                                fprintf('Initializing layerConnect connections %d->%d\n',fromlayer,tolayer);
                                obj.layerConnect(tolayer,fromlayer)=1;
                            end
                        case 'next'
                            fprintf('Initializing layerConnect to adjacent connections %d->%d\n',fromlayer,tolayer);
                            fromlayer = tolayer-1;
                            obj.layerConnect(tolayer,fromlayer)=1;
                    end
                end
            end            
            
            % default input connect
            % 'all'; 'firstlast': input->first layer, input->last layer
            if isempty(obj.inputConnect) || (numel(obj.inputConnect) ~= obj.nbrofLayers)
                obj.inputConnections = 'all';
                switch obj.inputConnections
                    case 'all'
                        fprintf('Initializing inputConnect to all %d forward layers\n',obj.nbrofLayers);
                        obj.inputConnect = ones(obj.nbrofLayers,1);
                    case 'firstlast'
                        fprintf('Initializing inputConnect to in->first, in->last layers\n');
                        obj.inputConnect = zeros(obj.nbrofLayers,1);
                        obj.inputConnect(1) = 1;
                        obj.inputConnect(obj.nbrofLayers) = 1;
                end
            end
            
            % default bias connect is for all layers
            if isempty(obj.biasConnect) || (numel(obj.biasConnect) ~= obj.nbrofLayers)
                fprintf('Including biasConnect for all %d layers\n', obj.nbrofLayers);
                obj.biasConnect = ones(obj.nbrofLayers,1);
            end    
                        
            switch obj.initFcn
                % weights already initialized
                case 'previous'
                    if all( isempty(obj.LayerWeights) ) || ...
                            any( isempty(obj.bias) ) || ...
                            all( isempty(obj.InputWeights) )
                        error('change obj.initFcn=%s, since previous weights are empty',obj.initFcn);
                    end
                    for layer=1:obj.nbrofLayers                        
                        DeltaIW{layer} = zeros(size(obj.InputWeights{layer}));
                        Deltab{layer} = zeros(size(obj.bias{layer}));                        
                        for fromlayer = 1:layer-1
                            DeltaLW{layer,fromlayer} = zeros(size(obj.LayerWeights{layer,fromlayer}));
                        end                        
                    end
                otherwise
                    for layer=1:obj.nbrofLayers
                        % ---------------------------------------------------------
                        % W A R N I N G
                        % W*x used here (and in neural nets) instead of W'*x (ABF)
                        % since y = W*x, the dim of W is (out x in) NOT (in x out)
                        
                        % initialize the input weights and bias
                        % and bias for each layer
                        nbrofNeurons = obj.nbrofUnits(2,layer);
                        nbrofIn = obj.nbrofUnits(1,1);
                        [w,b] = obj.init(nbrofNeurons,nbrofIn);
                        if obj.inputConnect(layer)==0, w= []; end
                        obj.InputWeights{layer} = w;
                        DeltaIW{layer} = zeros(size(w));

                        if obj.biasConnect(layer)==0, b=[]; end
                        obj.bias{layer} = b;
                        Deltab{layer} = zeros(size(b));
                        
                        for fromlayer = 1:layer-1
                            nbrofNeurons = obj.nbrofUnits(2,layer);
                            nbrofIn = obj.nbrofUnits(2,fromlayer);
                            [w,~] = obj.init(nbrofNeurons,nbrofIn);
                            if obj.layerConnect(layer,fromlayer)==0, w=[]; end
                            obj.LayerWeights{layer,fromlayer} = w;
                            DeltaLW{layer,fromlayer} = zeros(size(w));                                                       
                        end                        
                    end
            end
                        
            obj.nbrofWeights = zeros(1,obj.nbrofLayers);
            for layer=1:obj.nbrofLayers
                % input weights coming into this layer            
                num = numel(obj.InputWeights{layer});                     
                for fromlayer = 1:layer-1
                    % weights from other layers
                    num = num + numel(obj.LayerWeights{layer,fromlayer});
                end
                obj.nbrofWeights(layer) = num;
            end
            obj.nbrofParameters = sum(obj.nbrofWeights+obj.nbrofBias);
            
            switch obj.trainFcn
                case {'trainlm','trainbr'}
                    Jac = zeros(obj.nbrofParameters,nbrofSamplesinBatch);
                    % Hessian and Jac*error are derived from Jac and don't
                    % need to be pre allocated
                    %Hessian = zeros(nbrofParameters,nbrofParameters);
                    %jace = zeros(nbrofParameters,1);                                        
                    [sensitivityf] = deal(cell(obj.nbrofLayers,1));                                        
                    [JacIW, Jacb] = deal(cell(obj.nbrofLayers,1));
                    [JacLW] = deal(cell(obj.nbrofLayers,obj.nbrofLayers));                   
                    
                    for layer=1:obj.nbrofLayers
                        JacIW{layer} = zeros([size(DeltaIW{layer}),nbrofSamplesinBatch]);
                        Jacb{layer} = zeros([size(Deltab{layer}),nbrofSamplesinBatch]);                        
                        for fromlayer = 1:layer-1
                            JacLW{layer,fromlayer} = ...
                                zeros([size(DeltaLW{layer,fromlayer}),nbrofSamplesinBatch]);                            
                        end                        
                    end
            end
            
            % need a way to vectorize the weights for storage, and for
            % Jacobian calculations in the LM case
            [obj] = getlayerinds(obj);                    
            
            % setup a FIFO buffer for storing past max_fail weights so
            % that we can scroll back when validate data mse increases
            bIWLW = Weights_to_vec(obj,obj.bias,obj.InputWeights,obj.LayerWeights);
            obj.WeightsBuffer = zeros(numel(bIWLW),obj.max_fail+1);                    
            obj.WeightsBuffer(:,1) = bIWLW;
       
            % update the weights over the epochs
            [Msetrain,Msetest,Msevalidate] = deal(-1*ones(1,obj.nbrofEpochs));
            %lrate = obj.lrate;
            %disp('input learning rate is not being used, set in gradient directory');     
            [Gradient,Mu,Gamma,Ew] = deal(nan(1,obj.nbrofEpochs));
            
            epoch = 0; 
            keeptraining = 1; 
            testfail = 0;
            tstart = tic;            
            
            % initializing mse to inf even though not true, first step            
            % always takes place
            [best.msetst,best.msetrn,best.msevl,best.perftrn] = deal(Inf);
            best.bIWLW=bIWLW;
            
            % Regularization parameters needed for trainlm and trainbr
            totalnbrofParameters = (2-isreal(bIWLW))*obj.nbrofParameters;
            obj.gamma = totalnbrofParameters;
            
            %--------------S T A R T    E P O C H    L O O P --------------
            %--------------------------------------------------------------
            while keeptraining
                epoch=epoch+1;                
                printthisepoch = (epoch/obj.printmseinEpochs == floor(epoch/obj.printmseinEpochs));
                
                 % pick a batch for this epoch for training or keep it fixed
                % throughout (@todo: could pull fixed out of the loop but
                % more readable to do it here)
                switch obj.batchtype
                    case 'randperm'
                        batchtrain = randperm(nbrofSamples,nbrofSamplesinBatch);
                    case 'fixed'
                        batchtrain = 1:nbrofSamplesinBatch;                        
                    case 'index'
                        batchtrain = obj.trainInd; nbrofSamplesinBatch = length(obj.trainInd);
                        batchtest = obj.testInd; nbrofSamplesinTest = length(obj.testInd);
                        batchvalidate = obj.valInd; nbrofSamplesinValidate = length(obj.valInd);
                end                
                switch obj.batchtype
                    case {'randperm','fixed'}
                        batchleft = setdiff(1:nbrofSamples,batchtrain);
                        batchtest = batchleft(1:nbrofSamplesinTest);
                        batchvalidate = batchleft(nbrofSamplesinTest + (1:nbrofSamplesinValidate));
                end
                
                % training, testing, and validation
                trn_normalized = out_normalized(:,batchtrain);
                trn = out(:,batchtrain);                
                tst_normalized = out_normalized(:,batchtest);
                tst = out(:,batchtest);                
                vl_normalized = out_normalized(:,batchvalidate);
                vl = out(:,batchvalidate);
                
                
                % teacher error is a form of randomization instead of using
                % different training/testing/validation at each epoch
                if 0 %epoch/10 == floor(epoch/10)
                    %----------------------------------------------------------
                    % introduce teacher error at 10 x smallest number
                    % "Proposal of relative-minimization learning for behavior
                    % stabilization of complex-valued recurrent neural
                    % networks" by Akira Hirose, Hirofumi Onishi
                    teachererrorsize = 1e4*eps;
                    teachererror = teachererrorsize*(2*rand(size(in(:,batchtrain)))-1);
                    if any(imag(in(:)))
                        teachererror = teachererror + ...
                            1i*teachererrorsize*(2*rand(size(in(:,batchtrain)))-1);
                    end
                    %----------------------------------------------------------
                    fprintf('Using teacher errors +/- %e for robust learning\n',teachererrorsize);
                else
                    teachererror = 0;
                end

                % evaluate network and obtain gradients and intermediate values
                % y{end} is the output                                                
                [curr,n,a0,a,fdot] = obj.test( in(:,batchtrain) + teachererror);                
                curr_normalized = a{obj.nbrofLayers};
                
                % mean squared error in the unnormalized
                % not taking 1/2 of the square 
                % normalized would be y{obj.nbrofLayers} - out_normalized
                msecurr = mean( abs(curr(:)-trn(:)).^2 );
                ew = (bIWLW'*bIWLW);
                ed = msecurr*1;%nbrofSamplesinBatch;
                
                if epoch==1
                    switch obj.trainFcn
                        case 'trainbr'
                            if ew < 100*eps
                                obj.alpha = 1;
                            else
                                obj.alpha = obj.gamma/(2 * ew);
                            end
                            obj.beta = (nbrofSamplesinBatch*nbrofInUnits - obj.gamma)/(2 * ed);
                            if (obj.beta<=0), obj.beta = 1/2; end
                    end
                end                
                perfcurr = obj.beta*ed + obj.alpha*ew;
                
                if epoch==1
                    best.msetrn = msecurr;
                    best.perftrn =  perfcurr;
                end
                
                if obj.debugPlots && printthisepoch                                        
                    figure(901);clf;
                    subplot(211)
                    num = min(20,nbrofSamplesinBatch);
                    plot(curr(:,1:num),'.-'); hold on;
                    plot(trn(:,1:num),'o-');
                    ylim([-10 10]);
                    subplot(212)
                    plot(curr_normalized(1:num),'.-'); hold on;
                    plot(trn_normalized(1:num),'o-');
                    ylim([-1 1]);
                end
                
                
                %----------------------------------------------------------
                % backpropagate to get all layers                
                splitrealimag = zeros(1,obj.nbrofLayers);
                gradientacc = 0;
                for layer=obj.nbrofLayers:-1:1                             
                    % sensitivity is the gradient at the middle of the layer
                    % sensitivity = d{cost}/d{xn} called "sensitivity"
                    % 
                    [sensitivity{layer},sensitivityf{layer}] = ...
                        deal(zeros( obj.nbrofUnits(2,layer), nbrofSamplesinBatch));                   
                    
                    splitrealimag(layer)=0; 
                    if isstruct(fdot{layer}), splitrealimag(layer) = 1; end
                                        
                    if layer==obj.nbrofLayers
                        % for output layer, using mean-squared error
                        % w = w - mu (y-d) f'(net*) x*                        

                        % have to include the dx/dy = 1/gain on the output
                        % since mse is computed there
                        % @todo: a better way is to make a constant output weighting node
                        err = curr-trn;
                        if splitrealimag(layer)            
                            fdot_times_outputmap_derivative = ...
                                obj.dounrealifyfn(derivative_outputmap * obj.dorealifyfn(fdot{layer}.real + 1i*fdot{layer}.imag));                            
                            sensitivity{layer} = real(err) .* real(fdot_times_outputmap_derivative) + ...
                                1i* imag(err).* imag(fdot_times_outputmap_derivative);                                                        
                            sensitivityf{layer} = fdot_times_outputmap_derivative;                            
                        else  
                            fdot_times_outputmap_derivative = obj.dounrealifyfn(derivative_outputmap * obj.dorealifyfn(fdot{layer}));                            
                            sensitivity{layer} = err .* conj( fdot_times_outputmap_derivative );                            
                            sensitivityf{layer} = conj( fdot_times_outputmap_derivative );         
                        end                        
                    else                                                
                        % factor out the fdot from the r loop
                        % as in fdot  * sum{ sensitivity{tolayer}*W }   
                        %                \-----LW_sensitivity-------/                        
                        if splitrealimag(layer)
                            LW_sensitivity.real = 0; LW_sensitivity.imag = 0;
                            LW_sensitivityf.real = 0; LW_sensitivityf.imag = 0;
                        else
                            LW_sensitivity = 0;
                            LW_sensitivityf = 0;
                        end                        
                        for tolayer = layer+1:obj.nbrofLayers                            
                            LWtolayer = obj.LayerWeights{tolayer,layer};                                      
                            % if no layer weight, no contribution
                            if ~isempty(LWtolayer)                                
                                % in backprop, sensitivity to layer
                                stolayer = sensitivity{tolayer};
                                sftolayer = sensitivityf{tolayer};
                                
                                if splitrealimag(layer)
                                    % for 2-layer derivation, see equation(17)
                                    % "Extension of the BackPropagation
                                    % algorithm to complex numbers" by Tohru Nitta
                                    LW_sensitivity.real = LW_sensitivity.real + ...
                                        (  (real(LWtolayer).' * real(stolayer)) +...
                                        (imag(LWtolayer).' * imag(stolayer)) );
                                    LW_sensitivity.imag = LW_sensitivity.imag + ...
                                        (  (imag(LWtolayer).' * real(stolayer)) +...
                                        -1*(real(LWtolayer).' * imag(stolayer) ) );
                                    
                                    LW_sensitivityf.real = LW_sensitivityf.real + ...
                                        (  (real(LWtolayer).' * real(sftolayer)) +...
                                        (imag(LWtolayer).' * imag(sftolayer)) );
                                    LW_sensitivityf.imag = LW_sensitivityf.imag + ...
                                        (  (imag(LWtolayer).' * real(sftolayer)) +...
                                        -1*(real(LWtolayer).' * imag(sftolayer)) );
                                    
                                else
                                    LW_sensitivity = LW_sensitivity + ...
                                        ( LWtolayer.' * stolayer);
                                    LW_sensitivityf = LW_sensitivityf + ...
                                        ( LWtolayer.' * sftolayer);
                                end
                            end
                        end
                        
                        fdotlayer = fdot{layer};
                        if splitrealimag(layer)
                            sensitivity{layer} = fdotlayer.real.*LW_sensitivity.real +...
                                -1i* fdotlayer.imag .* LW_sensitivity.imag;
                            sensitivityf{layer} = fdotlayer.real.*LW_sensitivityf.real +...
                                -1i* fdotlayer.imag .* LW_sensitivityf.imag;                            
                        else
                            sensitivity{layer} = conj(fdotlayer).*LW_sensitivity;
                            sensitivityf{layer} = conj(fdotlayer).*LW_sensitivityf;                   
                        end
                    end % if nn=nbrofLayers, i.e. last layer

                    if obj.biasConnect(layer)
                    % bias added at each layer
                    Deltab{layer} = sum( sensitivity{layer},2)/nbrofSamplesinBatch;         
                    end
                    
                    if obj.inputConnect(layer)
                    % weights applied to input 1->2, 1->3,...
                    DeltaIW{layer} = sensitivity{layer}*a0'/nbrofSamplesinBatch;
                    end
                    
                    % weights applied to previous layers 2->3,2->4, etc...
                    for fromlayer = 1:layer-1
                        if obj.layerConnect(layer,fromlayer)
                        % weighted average of sensitivity with input measurements                        
                        inputtolayer = a{fromlayer};
                        DeltaLW{layer,fromlayer} = sensitivity{layer}*inputtolayer'/nbrofSamplesinBatch;
                        end
                    end
                    
                    gradientacc = gradientacc + ...
                        sum(abs(Deltab{layer}(:)).^2) + ...
                        sum(abs(DeltaIW{layer}(:)).^2);
                    for fromlayer = 1:layer-1
                        gradientacc = gradientacc + ...
                            sum(abs(DeltaLW{layer,fromlayer}(:)).^2);
                    end
                    
                    % Jacobian dcost/dparameters
                    for ll=1:obj.nbrofUnits(2,layer)
                        if obj.biasConnect(layer)
                            Jacb{layer}(ll,:,:) = sensitivityf{layer}(ll,:);
                        end
                        if obj.inputConnect(layer)
                            JacIW{layer}(ll,:,:) = bsxfun(@times,conj(a0),sensitivityf{layer}(ll,:));
                        end
                        for fromlayer = 1:layer-1
                            if obj.layerConnect(layer,fromlayer)
                                % weighted average of sensitivity with input measurements
                                inputtolayer = a{fromlayer};
                                JacLW{layer,fromlayer}(ll,:,:) =  bsxfun(@times,conj(inputtolayer),sensitivityf{layer}(ll,:));
                            end
                        end
                    end
                    
                    if obj.debugPlots && printthisepoch
                        figure(1023);
                        subplot(obj.nbrofLayers,1,layer)
                        imagesc(real(DeltaLW{layer+1,layer})); colorbar;
                    end
                    
                end
                gradientacc = sqrt( gradientacc );  
                
                %for linear single neuron, no output mapping
                % DeltaW = err_times_outputmap_derivative .* conj( yprime{nn} ) * ynnminus1'/nbrofSamplesinBatch;      
                % Jac = conj( yprime{nn} ) * conj(ynnminus1(1:end-1,mm)));
                % jace += transpose(Jac) * err_times_outputmap_derivative(:,mm);
                
                % update all the weight matrices (including bias weights)
                %
                switch obj.trainFcn
                    case {'trainlm','trainbr'}
                        % convert jacobian matrix portions into big matrix
                        Jac = Weights_to_vec(obj,Jacb,JacIW,JacLW);
                        Jac = Jac / sqrt(nbrofSamplesinBatch);
                        Hessian = (Jac*Jac');
                        ee = eye(size(Hessian));
                        
                        % Use the gradient way of calculating 
                        % jace = Jacobian * error, since this propagates 
                        % the error real,im parts correctly.  
                        % For the Hessian = Jacobian * Jacobian', the 
                        % LM computation is correct since this involves 
                        % the propagation of the function and current 
                        % weights derivatives and does not involve the 
                        % current error.                        
                        % Only use this line for real only case:
                        %jace = Jac*transpose(curr_normalized - trn_normalized);
                        % this line works for real and complex case:
                        jace = Weights_to_vec(obj,Deltab,DeltaIW,DeltaLW);                                                
                        
                        % to match MATLAB's calculations and use of mu, 
                        % include the 1/2 scaling - this is included in
                        % beta now
                        %Hessian = Hessian/2;
                        %jace = jace/2; 
                        
                        %--------------------------------------------------
                        %        D E B U G with MATLAB jacobian
                        % this is mostly code for deep numerical dive
                        %--------------------------------------------------                        
                        %if obj.debugCompare
                        %    debug_lm(obj,Jac,jace,Hessian,matlablayerweightinds,matlablayerbiasinds);
                        %end
                        
                        if obj.debugGradient
                            jacetocompare = WbWb_to_bIWLW(obj,obj.someRecord(epoch).jace);
                            Jactocompare = WbWb_to_bIWLW(obj,obj.someRecord(epoch).Jac);
                            fprintf('epoch %d ||gradient complexnet - gradient complexcascadenet|| %f\n',...
                                epoch,norm(jacetocompare - jace));
                            fprintf('epoch %d ||Jacobian complexnet - Jacobian complexcascadenet|| %f\n',...
                                epoch,norm(Jactocompare - Jac));
                        end                        
                        
                        % current weights
                        bIWLW = Weights_to_vec(obj,obj.bias,obj.InputWeights,obj.LayerWeights);        
                        totalnbrofParameters = (2-isreal(bIWLW))*obj.nbrofParameters;                        
                        ew = (bIWLW'*bIWLW);                        
                        ed = msecurr * 1;%nbrofSamplesinBatch;                                                
                        perfcurr = obj.beta * ed + obj.alpha * ew;                                                
                            
                        % generally, QR decomposition has better behavior
                        % than computing Hessian via Jac*Jac' but slower in
                        % software
                        USEQR = 0; 
                        % fast update is rank one updates to base qr(Jac')
                        FAST_UPDATE = 0;
                        if USEQR && FAST_UPDATE,  R = qr(Jac'); end
                        
                        trn = out(:,batchtrain);
                        msetrn = inf; perftrn=inf; numstep = 1;
                        while (perftrn>perfcurr) && (obj.mu<obj.mu_max)                                                    

                            if printthisepoch
                                % for debugging the LM optimization of loading
                                fprintf('epoch %d numstep %d mu %0.5f+alpha %0.5f\n',...
                                    epoch,numstep,obj.mu,obj.alpha);
                            end
                            
                            fprintf('epoch %d numstep %d mu %0.5f+alpha %0.5f\n',...
                                epoch,numstep,obj.mu,obj.alpha);
                            
                            % include Bayesian Regularization
                            mua = obj.mu + obj.alpha;
                                                                                            
                            %----------------------------------------------
                            % Blended Newton / Gradient
                            % (i.e. LevenBerg-Marquardt)
                            if USEQR==0
                                % loading the Hessian and regularizing
                                Hblend = obj.beta*Hessian + mua*ee;
                                
                                if isnan(rcond(Hblend))
                                    error('Condition number of blended Hessian is nan.  What did you do?');
                                end
                                deltabIWLW = Hblend\(obj.beta*jace + obj.alpha*bIWLW);
                            else
                                % use qr decomposition instead of Hessian which
                                % is obtained by squaring
                                if FAST_UPDATE
                                    R1 = triu(qrupdate_al( R, sqrt(mua)));
                                else
                                    Jact = [sqrt(obj.beta)*Jac sqrt(mua)*eye(obj.nbrofParameters)]';
                                    R1 = triu(qr(Jact,0));
                                end                                
                                deltabIWLW = R1\(R1'\(obj.beta*jace + obj.alpha*bIWLW));                                                                                                
                                % find extra step e based on residual error r
                                % does not generally help
                                %r = (obj.beta*jace + obj.alpha*WbWb) - Hblend*deltaWbWb;
                                %e = R1\(R1'\r);                               
                            end
                            %----------------------------------------------                            
                            
                            % convert vector to Weights and step in
                            % negative gradient direction
                            [Deltab,DeltaIW,DeltaLW] = vec_to_Weights(obj,deltabIWLW);                            
                            obj = updateWeights(obj,Deltab, DeltaIW, DeltaLW, -1);
                            bIWLW = bIWLW - deltabIWLW;
                                                        
                            % Check the mse for the update
                            curr = obj.test( in(:,batchtrain) );                            
                            msetrn = mean( abs(curr(:)-trn(:)).^2 );
                            ew = (bIWLW'*bIWLW);
                            ed = msetrn * 1;%nbrofSamplesinBatch;          
                            perftrn = obj.beta * ed + obj.alpha * ew;                            
                            
                            if isnan(deltabIWLW)
                                warning('epoch %d deltabIWLW has nan values',epoch);
                            end
                            numstep=numstep+1;                            

                            % incorrect comparison: msetrn>msecurr 
                            % instead, looking for best so far 
                            %if msetrn>best.msetrn
                            %if perftrn>best.perftrn  
                            if perftrn>perfcurr                                
                                obj.mu=obj.mu*obj.mu_inc; % pad the Hessian more
                                % undo the weight step
                                obj = updateWeights(obj,Deltab, DeltaIW, DeltaLW, +1);
                                bIWLW = bIWLW - deltabIWLW;
                            else
                                % as mu decreases, becomes Newton's method
                                obj.mu = max(obj.mu * obj.mu_dec,obj.mu_min);
                            end
                        end
                        
                        % Regularization update for gamma, beta, alpha
                        switch obj.trainFcn
                            case 'trainbr'                                
                                obj.gamma = totalnbrofParameters - obj.alpha*trace( inv(obj.beta*Hessian + obj.alpha*ee) );
                                if ew < 100*eps
                                    obj.alpha = 1;
                                else
                                    obj.alpha = obj.gamma/(2 * ew);
                                end
                                obj.beta = (nbrofSamplesinBatch*nbrofInUnits - obj.gamma)/(2 * ed);
                                if (obj.beta<=0), obj.beta = 1/2; end
                        end              
                    otherwise
                        thegradientfunction = str2func(obj.trainFcn);
                        try
                            for layer=1:obj.nbrofLayers
                                if obj.biasConnect(layer)
                                    [Deltab{layer},stateb{layer}] = ...
                                        thegradientfunction(Deltab{layer},stateb{layer});
                                    obj.bias{layer} = obj.bias{layer} - Deltab{layer};
                                end
                                if obj.inputConnect(layer)
                                    [DeltaIW{layer},stateIW{layer}] = ...
                                        thegradientfunction(DeltaIW{layer},stateIW{layer});
                                    obj.InputWeights{layer} = obj.InputWeights{layer} - DeltaIW{layer};
                                end
                                
                                for fromlayer = 1:layer-1
                                    if obj.layerConnect(layer,fromlayer)
                                        [DeltaLW{layer,fromlayer},stateLW{layer,fromlayer}] = ...
                                            thegradientfunction(DeltaLW{layer,fromlayer},stateLW{layer,fromlayer});
                                        obj.LayerWeights{layer,fromlayer} = ...
                                            obj.LayerWeights{layer,fromlayer} - DeltaLW{layer,fromlayer};
                                    end
                                end
                            end
                        catch
                            error('Unable to train using %s. Path must have ./gradient_descent',obj.trainFcn);
                        end
                end
                
                % if you want a history of the weights in vector form
                %obj.WeightsHistory(:,epoch) = Weights_to_vec(obj,obj.bias,obj.InputWeights,obj.LayerWeights);
                bIWLW = Weights_to_vec(obj,obj.bias,obj.InputWeights,obj.LayerWeights);
                obj.WeightsBuffer(:,1:end-1) = obj.WeightsBuffer(:,2:end);
                obj.WeightsBuffer(:,end) = bIWLW;
                                
                % Check the mses for the update
                curr = obj.test( in(:,batchtrain) );
                msetrn = mean( abs(curr(:)-trn(:)).^2 );            
                curr = obj.test( in(:,batchtest) );
                msetst = mean( abs(curr(:)-tst(:)).^2 );                                
                curr = obj.test( in(:,batchvalidate) );
                msevl = mean( abs(curr(:)-vl(:)).^2 );                                
                                                            
                ew = (bIWLW'*bIWLW);
                ed = msetrn * 1;%nbrofSamplesinBatch;                        
                perftrn = obj.beta * ed + obj.alpha * ew;
                
                Msetrain(epoch)= msetrn;
                Msetest(epoch) = msetst;
                Msevalidate(epoch) = msevl;
                Gradient(epoch) = gradientacc;
                Mu(epoch) = obj.mu;
                Gamma(epoch) = obj.gamma;
                Ew(epoch) = ew;                

                if numel(batchvalidate)
                    % if validation performance improves, update best network
                    % and clear validation failure count to allow for up
                    % down variation
                    if (msevl < best.msevl)                    
                        best.bIWLW = bIWLW;
                        best.msevl = msevl;
                        best.msetst = msetst;
                        best.msetrn = msetrn;                        
                        best.perftrn = perftrn;                                        
                        best.epoch = epoch;
                        testfail = 0;
                        % if validation performance got worse, increase 
                        % failure count and allow for up to max_fail
                    elseif (msevl > best.msevl)                    
                        testfail = testfail + 1;
                    end                    
                else
                    % if performance improved, update best performance
                    if (msetrn < best.msetrn)                    
                        best.bIWLW = bIWLW;
                        best.msevl = msevl;
                        best.msetst = msetst;
                        best.msetrn = msetrn;                        
                        best.perftrn = perftrn;                        
                        best.epoch = epoch;                        
                    end
                end
                
                epochtime = toc(tstart);
                                
                if printthisepoch
                    fprintf('epoch %d: msetrain %f msetest %f msevalidate %f grad %e\n',...
                        epoch,(msetrn),(msetst),(msevl),gradientacc);                    
                    fprintf('epoch %d: gamma %0.3f mu %0.5f alpa %0.3f, beta/nbrofSamplesinBatch %0.3f, ed %0.3f ew %0.3f\n',...
                            epoch,obj.gamma,obj.mu,obj.alpha,obj.beta/nbrofSamplesinBatch,ed,ew);                        
                end                                
                                
                kt.epoch = (epoch < obj.nbrofEpochs);
                kt.time = (epochtime < obj.maxtime);
                kt.mu = (obj.mu < obj.mu_max);
                kt.mse = (msetrn > obj.msedesired);
                kt.grad = (gradientacc > obj.min_grad);
                kt.fail = (testfail < obj.max_fail);
                keeptraining = all( struct2array(kt) );
                
            end % while keeptraining
            %----------------E N D    E P O C H    L O O P ----------------
            %--------------------------------------------------------------
                        
            fprintf('Scrolling back weights to best at epoch %d\n',best.epoch);
            [obj.bias,obj.InputWeights,obj.LayerWeights] = vec_to_Weights(obj,best.bIWLW);
            msetrn = best.msetrn;
            msetst = best.msetst;
            msevl = best.msevl;
            
            fprintf('time to train %0.3f sec, exit epoch %d: msetrain %f msetest %f msevalidate %f\n\n',...
                epochtime,epoch,(msetrn),(msetst),(msevl));
            fprintf('keeptraining flags (0 = exit condition reached):\n');
            disp(kt);
            
            if obj.performancePlots
                figure(231); clf;
                ha(1) = subplot(411);
                semilogy(Msetrain(1:epoch),'b.-','MarkerSize',20); hold on;
                semilogy(Msetest(1:epoch),'r+-','MarkerSize',4);
                semilogy(Msevalidate(1:epoch),'go-','MarkerSize',4);
                semilogy(1:epoch,msevl*ones(1,epoch),'.','MarkerSize',4)
                 legend('train (for determining weights)',...
                    'test (for reporting performance)',...
                    sprintf('validate (exit on %d increases from best)',obj.max_fail),...
                    'optimal weights','FontSize',12,'FontWeight','bold');
                grid minor;                
                ylabel('mean-squared error','FontSize',12,'FontWeight','bold');
                title('Trajectory of performance','FontSize',12,'FontWeight','bold');
                ha(2) = subplot(412);
                semilogy(Gradient(1:epoch),'.-','MarkerSize',20); ylabel('gradient','FontSize',12,'FontWeight','bold');
                grid minor;
                ha(3) = subplot(413);
                semilogy(Mu(1:epoch),'.-','MarkerSize',20); ylabel('mu','FontSize',12,'FontWeight','bold');
                grid minor;
                ha(4) = subplot(414);
                semilogy(Ew(1:epoch),'.-','MarkerSize',20); ylabel('||w||^2','FontSize',12,'FontWeight','bold');
                xlabel('epoch','FontSize',12,'FontWeight','bold');
                grid minor;
                linkaxes(ha,'x');
            end            
            
        end
        
        
    end
end

