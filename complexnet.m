classdef complexnet < handle
    % complexnet Feed-forward neural network with variety of
    % backpropagation approaches
    %
    % Swaroop Appadwedula Gr.64     % August 13,2020 in the time of COVID
    
    properties
        hiddenSize   % vector of hidden layer sizes, not including output
                     % empty when single layer                   
        layers       % cell array containing transferFcn spec, etc
                     % layers{ll}.transferFcn
        nbrofLayers  % number of layers hidden + 1 
        nbrofUnits   % 2xnbrofLayers matrix 
                     % with row1=input dim, row2=output dim
        nbrofWeights % count of #of weights and # bias in each layer
        nbrofBias
        
        totalnbrofParameters
        layerweightinds  % how weights and bias can be vectorized
        layerbiasinds        
        
        Weights      % cell array with the weights in each layer
        lrate        
        % keep weights as a vector
        WeightsHistory         % history of weights
        WeightsCircularBuffer  % circular buffer of last max_fail weights
        
        
        
        cascadeWeights 
        cascadeWeightsHistory         % history of weights
        cascadeWeightsCircularBuffer  % circular buffer of last max_fail weights
        
        
        
        

        trainFcn     % see options in gradient_descent directory
                     % Hessian-based 'trainlm'=Levenberg-Marquardt
              
        % Levenberg-Marquardt parameter for diagonal load to mix Hessian
        % step with gradient step
        mu
        mu_inc       % increment diagonal load when mse increases
        mu_dec
        mu_max       % bound the values of the diagonal load
        mu_min
                     
        initFcn      % 'randn','nguyen-widrow'

        minbatchsize
        batchtype    %'randperm','fixed'
        
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
        
        
        % for comparison with MATLAB feedforwardnet output
        debugCompare
        weightRecord
        jeRecord
        jjRecord        
        weightRecord2
        
    end
    
    properties (Constant)
        nbrofEpochsdefault = 1e3; % number of iterations picking a batch each time and running gradient
        beta1=0.9;         % Beta1 is the decay rate for the first moment
        beta2=0.999;       % Beta 2 is the decay rate for the second moment
        
        lrate_default = 1e-2; % initial step size for the gradient        

        initial_mu = 1e-3; % Hessian + mu * eye for Levenberg-Marquardt
               
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
        max_fail = 6;        % allow max_fail steps of increase in 
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
        function obj = complexnet(params)
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
                        
            if isfield(params,'lrate')
                obj.lrate= params.lrate; 
            else
                obj.lrate=obj.lrate_default;
            end
            
            if isfield(params,'mu')
                obj.mu = params.mu; 
            else
                obj.mu = obj.initial_mu;                
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
                    obj.Weights = deal(cell(1,obj.nbrofLayers));                    
            end
            
        end

        %----------------------------------------------------------        
        % single layer complex net
        % train on a single layer to verify derived expressions
        function [Weights,outhat] = trainsingle(obj,in,out)
            % trainsingle train a single layer network
            transferFcn = obj.layers{end}.transferFcn;
            fn = @(x) obj.(transferFcn)(x);
            
            [~,dz] = fn(1);
            if isstruct(dz), splitrealimag = 1; else, splitrealimag=0; end
                        
            [nbrofInUnits, nbrofSamples] = size(in);
            [nbrofOutUnits, ~] = size(out);
            
            
            if (nbrofSamples<nbrofInUnits)
                error('#samples %d must be larger than #features %d\n',...
                    nbrofSamples,nbrofInUnits);
            end
                                    
            % pick batch size number of sample if possible
            nbrofSamplesinBatch =  max(obj.batchsize_per_feature*nbrofInUnits,obj.minbatchsize);
            nbrofSamplesinBatch =  min(nbrofSamplesinBatch,nbrofSamples);
                        
            w = obj.crandn(nbrofOutUnits,nbrofInUnits);
            b = obj.crandn(1,1);
            mse = -1*ones(1,obj.nbrofEpochs);
            learningRate = obj.lrate;
            learningStep = 2;            
            batch = randperm(nbrofSamples,nbrofSamplesinBatch);     
            ee=eye(nbrofInUnits);
            for epoch=1:obj.nbrofEpochs                               
                % pick a batch for this epoch
                switch obj.batchtype
                    case 'randperm'
                        batch = randperm(nbrofSamples,nbrofSamplesinBatch);   
                    case 'fixed'
                        %batch stays the same, so mse can be examined for
                        %changes
                end
                
                y = in(:,batch);
                t = out(:,batch);           % desired y{nbrofLayers}                
                netval = w*y + b;
                [curr,gradf] = fn( netval );
                
                err = (curr - t);
                msecurr = sum(dot(err,err))/nbrofSamplesinBatch;                
                if splitrealimag
                    err_gradf = real(err).*gradf.real +1i*imag(err).*gradf.imag;
                    y_gradf = (gradf.real +1i*gradf.imag)*y'/nbrofSamplesinBatch;                 
                else                    
                    err_gradf = err.*conj(gradf);   
                    y_gradf = conj(gradf)*y'/nbrofSamplesinBatch;  
                end
                
                dw = err_gradf*y' / nbrofSamplesinBatch;
                
                switch obj.trainFcn
                    case 'trainlm'
                        % Levenberg Marquardt
                        for ll = 1:nbrofOutUnits
                            %dF = y_gradf(ll,:);
                            dF = bsxfun(@times,conj(y),err_gradf(ll,:))/nbrofSamplesinBatch;
                            Hblend = dF*dF' + learningRate*ee;
                            
                            %Hblend = learningRate * ee;
                            dw(ll,:) = dw(ll,:) / Hblend;
                        end
                        wnew = w - dw;                      
                    case 'Vanilla'
                        % Gradient descent
                        wnew = w - learningRate * dw;
                    otherwise
                        error('unrecognized trainFcn %s\n',obj.trainFcn);
                end
                
                d =  mean(  err,2 );
                bnew = b - d;                                   
                netvalnew = wnew*y + bnew;
                [new,~] = fn( netvalnew );
                err = (new - t);
                msenew = sum(dot(err,err))/nbrofSamplesinBatch;

                
                if msenew > msecurr
                    learningRate = learningRate * learningStep;
                    msenew = msecurr;
                    w = w; b = b;
                else                    
                    learningRate = learningRate / learningStep;
                    w = wnew; b = bnew;                    
                end                
                mse(epoch) = msenew;
                
                if (epoch/obj.printmseinEpochs == floor(epoch/obj.printmseinEpochs))
                    fprintf('%s mse(%d) %0.3f learningRate %0.7f \n',transferFcn,epoch,(mse(epoch)),learningRate);
                end
            end
            Weights = { w, b};
            obj.Weights = Weights;
            outhat = fn( w*in + b);
        end
        
            
        %----------------------------------------------------------
        % given an input, determine the network output and intermediate
        % terms
        function [out,n,a0,a,fdot] = test(obj,in)
            switch obj.domap
                case 0
                    % gain only
                    %in = bsxfun(@times,in,obj.inputSettings.gain);
                otherwise
                    % scale the input
                    ins = mapminmax.apply(obj.dorealifyfn(in),obj.inputSettings);
                    in = obj.dounrealifyfn(ins);            
            end
   
            % include bias vector for first layer
            % bias can be real since weights are complex
            % fyi: can zero out the bias by changing 1* to 0*
            biasvec = 1*ones(1,size(in,2));
            
            a0 = [in; biasvec];

            % used for derivative, but unnecessary since the corresponding
            % column is eliminated in the derivation of dc/dx
            zerovec = zeros(1,size(in,2));              
            
            ynnminus1 = a0;                        
            [n,a,fdot] = deal(cell(1,obj.nbrofLayers));
            for nn=1:obj.nbrofLayers
                % weight matrix
                W=obj.Weights{nn};
                
                % transfer function
                transferFcn = obj.layers{nn}.transferFcn;  % string
                activationfn = @(x) obj.(transferFcn)(x);  % handle
                
                % apply matrix of weights
                n{nn} = W*ynnminus1;     % x{n} = W{n}y{n-1}
                                
                % evaluate f(xn) and f'(xn)
                [a{nn},fdot{nn}] = activationfn(n{nn});
                
                % include a bias vector for next layer
                % final out y{nbrofLayers} does not have bias
                if nn < obj.nbrofLayers
                    % do not include the derivative of the bias w.r.t x{n}
                    % which is the zerovec
                    %yprime{nn} = [yprime{nn}; zerovec];
                    
                    % do include (additional row) the biasvec as part of y
                    a{nn} = [a{nn}; biasvec];
                end
                ynnminus1 = a{nn};
            end
               
            out_normalized = a{obj.nbrofLayers};
            switch obj.domap
                case 0
                    % gain only
                    %out = bsxfun(@rdivide,out_normalized,obj.outputSettings.gain);
                    out=out_normalized;
                otherwise
                    % network matches to -1,1 but output is back to original scale
                    out = mapminmax.reverse( obj.dorealifyfn(out_normalized),obj.outputSettings);
                    out = obj.dounrealifyfn(out);
            end
        end

        %------------------------------------------------------------------        
        % show the network when trained using train (not trainsingle)
        function print(obj)            
            for nn=1:obj.nbrofLayers
                fprintf('-----------------------------\n');
                if nn==1
                    fprintf('input layer\n'); 
                else
                    fprintf('hidden/output layer size %d\n',obj.hiddenSize(nn-1));
                end
                fprintf('W%d (out %d x in (%d+1 for bias) ) \n',nn, obj.nbrofUnits(2,nn), obj.nbrofUnits(1,nn)-1);                
                obj.Weights{nn}
                fprintf('\t%s\n',obj.layers{nn}.transferFcn);
                fprintf('-----------------------------\n');                
                fprintf('\t|\n');
            end
            fprintf('\toutput\n');
        
        end
        
        
       
        %------------------------------------------------------------------
        function obj = copynet(obj,net,IW,LW,b,weightRecord,jeRecord,jjRecord,weightRecord2)            
            % copy over matlab weights            

            if isempty(obj.Weights)
                obj.Weights = cell(1,obj.nbrofLayers);
            end
            
            if exist('IW','var') && exist('LW','var') && exist('b','var')
                %already assigned
            else
                IW = net.IW;  % these are the final values of weights
                LW = net.LW;
                b = net.b;
            end
                        
            if exist('weightRecord','var')
                % these are the weights in the weight record
                %[b,IW,LW] = separatewb(net,weightRecord{1});
                obj.weightRecord=weightRecord;
            end
            if exist('jeRecord','var')
                obj.jeRecord=jeRecord;
            end
            if exist('jjRecord','var')
                obj.jjRecord=jjRecord;
            end
            if exist('weightRecord2','var')
                % these are the weights in the weight record
                %[b,IW,LW] = separatewb(net,weightRecord{1});
                obj.weightRecord2=weightRecord2;
            end
            
            
            for ll = 1:obj.nbrofLayers                                                
                if ll==1
                    W = IW{1,1};
                else
                    W = LW{ll,ll-1};                    
                end
                bll = b{ll};
                                 
                if isempty(obj.Weights{ll})
                    fprintf('no weights, copying from net\n');
                    obj.Weights{ll} = [W bll];
                else  
                    Wb = obj.Weights{ll}; 
                    if any(size(Wb) ~= size([W bll]))
                        fprintf('not matching\n');
                    else
                        d = norm( Wb-[W bll]);
                        fprintf('Copied over weights %dx%d at layer %d from net || Wb net - Wb|| %f\n',...
                            size(Wb,1),size(Wb,2),ll,d)
                        obj.Weights{ll} = [W bll];
                    end
                end
            end
            obj.initFcn = 'previous';
            fprintf('Initialized obj.initFcn to %s for copied weights to be used as initial weights\n',obj.initFcn);
        end
        

        %------------------------------------------------------------------        
        % Jacobian contain derivatives of all parameters vectorized instead
        % of each set of weights in a cell
        % convention here is due to input being appended with ones at the
        % end, so that bias comes after weights
        % matlab convention is to have bibas first
        function [obj,matlablayerweightinds,matlablayerbiasinds] = getlayerinds(obj)
            cP = [0 cumsum(obj.nbrofWeights + obj.nbrofBias)];            
            %organize the terms in the Jacobian the same way as matlab
            [obj.layerweightinds,obj.layerbiasinds,...
                matlablayerweightinds,matlablayerbiasinds] = deal(cell(1,obj.nbrofLayers));
            for nn = 1:obj.nbrofLayers                
                % ordering is stacking one layer at a time from input
                
                % bias comes first for matlab (for debugging)
                matlablayerbiasinds{nn} =  (cP(nn)) + (1:obj.nbrofBias(nn));                                
                matlablayerweightinds{nn} = (cP(nn) + obj.nbrofBias(nn)) + (1:obj.nbrofWeights(nn));            
                
                % weight comes first
                obj.layerweightinds{nn} =  (cP(nn)) + (1:obj.nbrofWeights(nn));                                
                obj.layerbiasinds{nn} = (cP(nn) + obj.nbrofWeights(nn)) + (1:obj.nbrofBias(nn));            
            end  
        end        
        
        
        % inuput is Weights, output is WbWb
        function WbWb = Weights_to_vec(obj,Weights)
            WbWb = zeros(obj.totalnbrofParameters,1);
            for nn = 1:obj.nbrofLayers
                % Weights{nn} is output size x input size
                % input size includes +1 due to bias, so last column is the
                % bias
                WbWb([obj.layerweightinds{nn} obj.layerbiasinds{nn}]) = Weights{nn}(:);
            end            
        end
        
        % input vector is WbWb, output is Weights
        function Weights = vec_to_Weights(obj,WbWb)
            Weights = cell(1,obj.nbrofLayers);
            for nn=1:obj.nbrofLayers
                W = WbWb(obj.layerweightinds{nn},:);
                b = WbWb(obj.layerbiasinds{nn},:);
                if isvector(WbWb)
                    % single output
                    W = reshape(W,obj.nbrofUnits(2,nn),obj.nbrofUnits(1,nn)-1);
                    Weights{nn} = [W b];
                else
                    % multiple outputs
                    error('not handled yet - weights have wrong dimension');
                end               
            end
        end        
        
        function debug_lm(obj,Jac,jace,Hessian,matlablayerweightinds,matlablayerbiasinds)
            % from "comparenets.m"
            % sequence to allow for comparing nets:
            %
            % net=feedforwardnet()
            % net = configure(net,in,out)
            % save initial weights
            % train while saving Records of je,jj
            % copynet to copy the records
            %
            % matlab uses bias, weights, ... convention
            %[b,IW,LW] = separatewb(net,weightRecord{1});    
            epoch = 1;
            try
                wr = obj.weightRecord{epoch};
                jer = obj.jeRecord{epoch};
                jjr = obj.jjRecord{epoch};
                wr2 = obj.weightRecord2{epoch};
                
                Weightsmatlab = cell(1,obj.nbrofLayers);
                Weightsmatlab2 = cell(1,obj.nbrofLayers);
                matlablayersinds = []; layersinds = [];
                for nn=1:obj.nbrofLayers
                    dim_out = obj.nbrofUnits(2,nn);
                    dim_in = obj.nbrofUnits(1,nn)-1;
                    
                    b = wr(matlablayerbiasinds{nn});
                    w = wr(matlablayerweightinds{nn});
                    w = reshape( w, dim_out,dim_in);
                    Weightsmatlab{nn}=[w b];
                    
                    b = wr2(matlablayerbiasinds{nn});
                    w = wr2(matlablayerweightinds{nn});
                    w = reshape( w, dim_out,dim_in);
                    Weightsmatlab2{nn}=[w b];
                    
                    jeb = jer(matlablayerbiasinds{nn});
                    jew=jer(matlablayerweightinds{nn});
                    
                    % compare with existing
                    fprintf('layer%d cnet weights\n',nn);
                    obj.Weights{nn}
                    fprintf('layer%d net weights\n',nn);
                    Weightsmatlab{nn}
                    fprintf('layer%d net weights2 (after update)\n',nn);
                    Weightsmatlab2{nn}
                    
                    fprintf('layer%d [jew jace]\n',nn);
                    [jew jace(obj.layerweightinds{nn})]
                    fprintf('layer%d [jeb jace]\n',nn);
                    [jeb jace(obj.layerbiasinds{nn})]
                    
                    matlablayersinds = [ matlablayersinds matlablayerweightinds{nn} matlablayerbiasinds{nn}];
                    layersinds = [ layersinds obj.layerweightinds{nn} obj.layerbiasinds{nn}];
                end
                
                % convert from matlab's indexing to WbWb
                % and compare each element
                jjrtocompare = zeros(size(jjr));
                for ll = 1:obj.totalnbrofParameters
                    for mm = ll:obj.totalnbrofParameters
                        jjrtocompare(ll,mm) = jjr(matlablayersinds(ll),matlablayersinds(mm));
                        jjrtocompare(mm,ll) = jjr(matlablayersinds(mm),matlablayersinds(ll));
                    end
                end
                jertocompare = jer(matlablayersinds);
                fprintf('Jacobian * error\n');
                jace./jertocompare
                
                fprintf('Hessian\n');
                Hessian./jjrtocompare
                
                somescalar = mean(jace./jertocompare,'omitnan');
                somescalar2 = mean(Hessian(:)./jjrtocompare(:),'omitnan')
                
                fprintf('Ratio of my Hessian/ matlab Hessian [%e %e]\n',...
                    1-min(abs(Hessian(:)./jjrtocompare(:))),...
                    max(abs(Hessian(:)./jjrtocompare(:)))-1);
                fprintf('norm difference jace %e, Hessian %e\n',...
                    norm(jace/somescalar - jertocompare),...
                    norm(Hessian/somescalar2 - jjrtocompare));
                
                dWmatlab = (jjrtocompare + 1e-3 * eye(obj.totalnbrofParameters))\jertocompare;
                
                dW1 = (Hessian + 1e-3 * eye(obj.totalnbrofParameters))\jace;
                dW1s = (Hessian/somescalar2+ 1e-3 * eye(obj.totalnbrofParameters))\(jace/somescalar);
                
                % QR decomposition approach
                Jact = [Jac sqrt(1e-3)*eye(obj.totalnbrofParameters)]';
                R1 = triu(qr(Jact));
                dW1qr = R1\(R1'\jace);
                
                % iterate to get better - fails for now...
                A = (R1'*R1); %
                A = (Jact'*Jact);
                r = jace - A*dW1qr;
                e = R1\(R1'\r);
                dW1qrit = dW1qr + e;
                
                % min norm approach
                %dW1lsq = lsqr(Jact'*Jact,jace);
                %norm(dW1lsq - dWmatlab)
                dW1lsq = lsqminnorm(Jact'*Jact,jace);
                
                disp('no scaling d(cnet - matlab)');
                norm(dW1 - dWmatlab)
                disp('scaled d(cnet - matlab)');
                norm(dW1s - dWmatlab)
                disp('scaled d(cnet qr - matlab)');
                norm(dW1qr - dWmatlab)
                
                disp('scaled d(cnet qr  iteration - matlab)');
                norm(dW1qrit - dWmatlab)
                
                disp('d(cnet minnorm - matlab)');
                norm(dW1lsq - dWmatlab)
                
                %{
                % svd approach
                [~,S,V] = svd(Jact);
                svec = diag(S); svec2 = 1./(svec.*conj(svec));
                dW1svd = (V*diag(svec2)*V')*jace;
                disp('d(cnet svd - matlab)');
                norm(dW1svd - dWmatlab)
                %}
                
            catch
                Jac
                Hessian
            end
            keyboard;
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
            
            % allocate space for gradients
            % sensitivity{n} = dcost/dx{n} vector of dimension x{n}
            % DeltaW{n} = dcost/dWeights{n} matrix of dimension W{n}
            % state is cell array with state of gradient and hessian approx
            % there is redundancy here since each layer has it's own state
            % that has rate parameters that are same - not a big deal
            [sensitivity,DeltaW,state] = deal(cell(1,obj.nbrofLayers));
            
            switch obj.trainFcn
                case 'trainlm'
                    [Deltaf] = deal(cell(1,obj.nbrofLayers));
            end            
            
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
            
            
            % once a input is provided, dimensions can be determined
            % initialize the weights
            % this matrix indicates the input (row1) and output (row2)
            % dimensions            
            obj.nbrofUnits(1,:) = [nbrofInUnits  nbrOfNeuronsInEachHiddenLayer];  % input size
            obj.nbrofUnits(2,:) = [nbrOfNeuronsInEachHiddenLayer nbrofOutUnits];  % output size            

            % count of number of weights and bias at each layer
            obj.nbrofWeights = obj.nbrofUnits(1,:).* obj.nbrofUnits(2,:);
            obj.nbrofBias = obj.nbrofUnits(2,:);
            
            % include a bias to allow for functions away from zero
            % bias changes the input dimension since weigths are applied to
            % bias as well, but it does not change output dimension
            obj.nbrofUnits(1,1:end) =  obj.nbrofUnits(1,1:end) + 1;    % all input sizes of layers incremented                   

            for layer=1:obj.nbrofLayers                    
                % warning: W*x used here instead of W'*x
                % since y = W*x, the dimensions of W are out x in 
                % NOT in x out                                
                switch obj.initFcn
                    %-----------------------------
                    % complex initializations
                    case 'crandn'
                        obj.Weights{layer} = 0.01*obj.crandn( obj.nbrofUnits(2,layer), obj.nbrofUnits(1,layer) ); 
                     case 'crands'
                        obj.Weights{layer} = rands( obj.nbrofUnits(2,layer), obj.nbrofUnits(1,layer) ) + ...
                            1i*rands( obj.nbrofUnits(2,layer), obj.nbrofUnits(1,layer) ); 
                       
                    case 'c-nguyen-widrow'
                        %Matlab's Nguyen-Widrow Algorithm
                        obj.Weights{layer} = zeros(  obj.nbrofUnits(2,layer), obj.nbrofUnits(1,layer) );
                        activeregionofactivation = [-2 2];  % tansig('active');
                        mapregion = [-1 1];
                        nbrofNeurons = obj.nbrofUnits(2,layer);
                        nbrofIn = obj.nbrofUnits(1,layer)-1;
                        mapregions = repmat(mapregion,nbrofIn,1);
                        [wr,br]=mycalcnw(mapregions,nbrofNeurons,activeregionofactivation);
                        [wi,bi]=mycalcnw(mapregions,nbrofNeurons,activeregionofactivation);                        
                        obj.Weights{layer} = [wr+1i*wi br+1i*bi];                        
                        
                    %-----------------------------
                    % non-complex initializations                                        
                    case 'randn'
                        obj.Weights{layer} = 0.01*randn( obj.nbrofUnits(2,layer), obj.nbrofUnits(1,layer) );                        

                    case 'nguyen-widrow'
                        %Matlab's Nguyen-Widrow Algorithm
                        obj.Weights{layer} = zeros(  obj.nbrofUnits(2,layer), obj.nbrofUnits(1,layer) );
                        activeregionofactivation = [-2 2];  % tansig('active');
                        mapregion = [-1 1];
                        nbrofNeurons = obj.nbrofUnits(2,layer);
                        nbrofIn = obj.nbrofUnits(1,layer)-1;
                        mapregions = repmat(mapregion,nbrofIn,1);
                        [w,b]=mycalcnw(mapregions,nbrofNeurons,activeregionofactivation);
                        obj.Weights{layer} = [w b];
                        
                    % weights already initialized                        
                    case 'previous'
                        if isempty(obj.Weights)
                            error('change obj.initFcn=%s, since previous weights are empty',obj.initFcn);
                        end
                        if any( size(obj.Weights{layer})~= [obj.nbrofUnits(2,layer), obj.nbrofUnits(1,layer)])
                            fprintf('layer%d Weights %d x %d ~= dimensions %d x %d\n',layer,...
                                size(obj.Weights{layer},1),size(obj.Weights{layer},2),...
                                obj.nbrofUnits(2,layer), obj.nbrofUnits(1,layer));
                            error('change obj.initFcn=%s, since previous weights dimensions not matching',obj.initFcn);                            
                        end
                        
                        % example debugging options
                        
                        %warning('setting all weights to 1 for debug')
                        %obj.Weights{nn} = ones(size(obj.Weights{nn}));
                        
                        %warning('setting weights for a special case')
                        %switch nn
                        %    case 2, obj.Weights{2}= [ obj.weightRecord{1}(5) obj.weightRecord{1}(4)];
                        %    case 1, obj.Weights{1}= [ obj.weightRecord{1}(2:3).' obj.weightRecord{1}(1)];
                        %end                
                    otherwise
                        error('unknown initFcn %s for weights',obj.initFcn);
                end
                
                %{
                % example debugging options
                switch obj.initFcn
                    case 'previous'
                    otherwise
                        fprintf('setting %d bias weights to zero for layer %d to initialize\n',obj.nbrofUnits(2,nn),nn);
                        obj.Weights{nn}(:,end)=0;
                end
                %}
                
                [DeltaW{layer}] = ...
                    deal(zeros( obj.nbrofUnits(2,layer), obj.nbrofUnits(1,layer) ));   
                
                switch obj.trainFcn            
                    case 'trainlm'            
                        % Deltaf matrices are due to weights only 
                        if layer==obj.nbrofLayers
                            Deltaf{layer} = zeros(nbrofOutUnits,nbrofOutUnits,nbrofSamplesinBatch);
                        else
                            Deltaf{layer} = zeros(obj.nbrofUnits(2,layer),nbrofOutUnits,nbrofSamplesinBatch);                            
                        end
                end
            end
            
            obj.totalnbrofParameters = sum(obj.nbrofWeights+obj.nbrofBias); 
            switch obj.trainFcn
                case 'trainlm'
                    Jac = zeros(obj.totalnbrofParameters,nbrofSamplesinBatch*nbrofOutUnits);
                    % Hessian and Jac*error are derived from Jac and don't
                    % need to be pre allocated
                    %Hessian = zeros(totalnbrofParameters,totalnbrofParameters);
                    %jace = zeros(totalnbrofParameters,1);                                        
                    [sensitivityf,DeltaF] = deal(cell(1,obj.nbrofLayers));                    
            end
            
            % need a way to vectorize the weights for storage, and for
            % Jacobian calculations in the LM case
            [obj,matlablayerweightinds,matlablayerbiasinds] = ...
                getlayerinds(obj);                    
            
            % setup a circular buffer for storing past max_fail weights so
            % that we can scroll back when validate data mse increases
            WbWb = Weights_to_vec(obj,obj.Weights);
            obj.WeightsCircularBuffer = zeros(numel(WbWb),obj.max_fail+1);                    
            obj.WeightsCircularBuffer(:,1) = WbWb;

            %{
            % for complex nets that get stuck in local min, use random
            % search algorithms for first 500 epochs  
            curr= obj.test( in);                        
            mse = mean( abs(curr(:)-out(:)).^2 );
            includeimag = any(imag(WbWb(:)));
            msepre = nan(1,500);
            for epochpre=1:500              
                if includeimag
                    dWbWb = 0.01*obj.crandn(size(WbWb,1),size(WbWb,2));
                else
                    dWbWb = 0.01*randn(size(WbWb,1),size(WbWb,2));
                end
                WbWbcurr = dWbWb;
                obj.Weights = vec_to_Weights(obj,WbWbcurr);
                                
                curr= obj.test( in);                        
                msecurr = mean( abs(curr(:)-out(:)).^2 );
                p(epochpre) = min(1,exp(-2*(msecurr-mse)/mse));
                if msecurr<mse, p(epochpre)=1; else p(epochpre)=0; end
                
                fprintf('epochpre %d mse %e msecurr %e p %f\n',epochpre,mse,msecurr,p(epochpre));
                if rand(1)<p(epochpre)
                    % accept new weights with probability p
                    mse = msecurr;
                    WbWb = WbWbcurr;
                    % obj.Weights already set
                else
                    % do not accept new weights, reset to previous
                    obj.Weights = vec_to_Weights(obj,WbWb);
                    % mse, WbWb not updated to curr
                end                
                msepre(epochpre) = mse;
            end
            figure(999); 
            subplot(211);plot(msepre);
            subplot(212); plot(p,'.')
            %}
            
            % update the weights over the epochs
            [msetrain,msetest,msevalidate] = deal(-1*ones(1,obj.nbrofEpochs));
            %lrate = obj.lrate;
            %disp('input learning rate is not being used, set in gradient directory');     
                        
            epoch = 0; 
            keeptraining = 1; 
            testfail = 0;
            tstart = tic;            
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
                end                
                batchleft = setdiff(1:nbrofSamples,batchtrain);                
                batchtest = batchleft(1:nbrofSamplesinTest);                
                batchvalidate = batchleft(nbrofSamplesinTest + (1:nbrofSamplesinValidate));
                
                % training, testing, and validation picks
                trn_normalized = out_normalized(:,batchtrain);
                trn = out(:,batchtrain);                
                tst_normalized = out_normalized(:,batchtest);
                tst = out(:,batchtest);                
                vl_normalized = out_normalized(:,batchvalidate);
                vl = out(:,batchvalidate);
                
                % teacher errors can help, need to pick error size relative
                % to desired mse floor which may be tricky
                if epoch/10 == floor(epoch/10)
                    %----------------------------------------------------------
                    % introduce teacher error at ??? x smallest number
                    % "Proposal of relative-minimization learning for behavior
                    % stabilization of complex-valued recurrent neural
                    % networks" by Akira Hirose, Hirofumi Onishi
                    teachererrorsize = 10000*eps;
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
                % B A C K   P R O P A G A T I O N
                %----------------------------------------------------------
                % start from last layer and recursively propagate gradient
                
                % nonlinearity is either split re/im or acts on both same
                splitrealimag = zeros(1,obj.nbrofLayers);
                
                % keep track of accumulated gradient for stopping criteria
                gradientacc = 0;
                
                for layer=obj.nbrofLayers:-1:1                             
                    % sensitivity is the gradient wrt n, at the middle of the layer
                    % sensitivity = d{error}^2/d{n}
                    % sensitivityf = d{error}/d{n}
                    
                    % split re/im derivatives are returned as .real,.imag
                    if isstruct(fdot{layer}), splitrealimag(layer) = 1; else, splitrealimag(layer)=0; end
                    
                    % a{layer} is the input to each layer
                    if layer==1 
                        aprev = a0; % no previous layer for first layer
                    else
                        aprev = a{layer-1}; 
                    end
                    
                    if layer==obj.nbrofLayers
                        % for output layer, mean-squared error allows for
                        % real and imag parts to be calculated sepaarately
                        %
                        % w = w - mu (y-d) f'(net*) x*                        
                        err = curr-trn;
                        if splitrealimag(layer)            
                            % fdot is the derivative of the output nonlinearity
                            % include the outputmap since it affects the error
                            fdot_times_outputmap_derivative = ...
                                obj.dounrealifyfn(derivative_outputmap * obj.dorealifyfn(fdot{layer}.real + 1i*fdot{layer}.imag));                            
                            sensitivity{layer} = real(err) .* real(fdot_times_outputmap_derivative) + ...
                                1i* imag(err).* imag(fdot_times_outputmap_derivative);                                                        
                        else  
                            fdot_times_outputmap_derivative = obj.dounrealifyfn(derivative_outputmap * obj.dorealifyfn(fdot{layer}));                            
                            sensitivity{layer} = err .* conj( fdot_times_outputmap_derivative );
                        end                                                
                                      
                        % compute sensitivityf = derror/dn for possibly
                        % multiple outputs.  each row is due to each output
                        % [ derror/dn      0          0     ]
                        % [    0        derror/dn     0     ]   
                        % [    0           0       derror/dn]
                        for outindex = 1:nbrofOutUnits
                            outputinds = (outindex-1)*nbrofSamplesinBatch + (1:nbrofSamplesinBatch);
                            if splitrealimag(layer)
                                sensitivityf{layer}(outindex,outputinds) = fdot_times_outputmap_derivative(outindex,:);                            
                            else
                                sensitivityf{layer}(outindex,outputinds) = conj( fdot_times_outputmap_derivative(outindex,:));         
                            end
                        end                        
                    else                        
                        % recursive steps for other layers 
                        % sensitivity{layer} = fdot{layer} * W{layer+1} * sensitivity{layer+1}
                        % sensitivityf{layer} = fdot{layer} * W{layer+1} * sensitivityf{layer+1}
                        sensitivity{layer} = ...
                            zeros( obj.nbrofUnits(2,layer), nbrofSamplesinBatch);                        
                        sensitivityf{layer} = ...
                            zeros( obj.nbrofUnits(2,layer), nbrofOutUnits*nbrofSamplesinBatch);
                                                
                        % last column of Weights are from the bias
                        % term for layer+1, which does not
                        % contribute to layer, hence 1:end-1
                        Wtolayer = obj.Weights{layer+1}(:,1:end-1);
                        fdotlayer = fdot{layer};                        

                        stolayer = sensitivity{layer+1};
                        if splitrealimag(layer)
                            % for 2-layer derivation, see equation(17)
                            % "Extension of the BackPropagation
                            % algorithm to complex numbers" by Tohru Nitta
                            fr = fdotlayer.real; fi = fdotlayer.imag;
                            sr = real(stolayer); si = imag(stolayer);
                            sensitivity{layer} = fr.* ...
                                (  (real(Wtolayer).' * sr) +...
                                (imag(Wtolayer).' * si) ) +...
                                -1i* fi .* ...
                                (  (imag(Wtolayer).' * sr) +...
                                -1*(real(Wtolayer).' * si ) );                            
                        else                            
                            sensitivity{layer} = conj(fdotlayer).* ( Wtolayer.' * stolayer);
                        end
                        
                        % for multiple outputs, need to loop through outputs
                        % easy way to think of this is that multiple
                        % outputs causes a fork at the last layer, which
                        % then requires keeping that many gradients
                        for outindex = 1:nbrofOutUnits
                            outputinds = (outindex-1)*nbrofSamplesinBatch + (1:nbrofSamplesinBatch);
                            sftolayer = sensitivityf{layer+1}(:,outputinds);
                            if splitrealimag(layer)
                                fr = fdotlayer.real; fi = fdotlayer.imag;
                                sfr = real(sftolayer); sfi = imag(sftolayer);                                
                                sensitivityf{layer}(:,outputinds) = fr.* ...
                                    (  (real(Wtolayer).' * sfr) +...
                                    (imag(Wtolayer).' * sfi) ) +...
                                    -1i* fi .* ...
                                    (  (imag(Wtolayer).' * sfr) +...
                                    -1*(real(Wtolayer).' * sfi ) );
                            else
                                sensitivityf{layer}(:,outputinds) = conj(fdotlayer).* ( Wtolayer.' * sftolayer);
                            end
                        end
                        
                    end % if layer=nbrofLayers, i.e. last layer
                    
                    % obj.nbrofUnits(2,nn) x obj.nbrofUnits(1,nn)
                    % weighted sum of sensitivity with a{layer-1}
                    DeltaW{layer} = sensitivity{layer}*aprev'/nbrofSamplesinBatch;

                    % Jacobian (DeltaF) terms are obtained as outer product
                    % of input a with sensitivityf
                    for outindex = 1:nbrofOutUnits
                        outputinds = (outindex-1)*nbrofSamplesinBatch + (1:nbrofSamplesinBatch);                        
                        for layeroutputindex = 1:obj.nbrofUnits(2,layer)
                            dF = zeros(obj.nbrofUnits(1,layer),nbrofSamplesinBatch);
                            ap = aprev(1:end-1,:);
                            sf = sensitivityf{layer}(layeroutputindex,outputinds);
 
                            % Jacobian terms for weights and bias
                            dF(1:end-1,:) = bsxfun(@times,conj(ap),sf);                            
                            dF(end,:) = sf;
                            
                            DeltaF{layer}(layeroutputindex,:,outputinds) = dF;
                        end
                    end
                    
                    if obj.debugPlots && printthisepoch
                        figure(1023);
                        subplot(obj.nbrofLayers,1,layer)
                        imagesc(real(DeltaW{layer})); colorbar;
                    end
                    
                    % accumulate the gradient
                    gradientacc = gradientacc + sum(abs(DeltaW{layer}(:)).^2);
                end
                gradientacc = sqrt( gradientacc );  
                
                
                %----------------------------------------------------------
                % W E I G H T    U P D A T E
                %----------------------------------------------------------                
                % update all the weight matrices (including bias weights)
                %
                switch obj.trainFcn
                    case 'trainlm'
                        % VECTORIZE the Jacobian and the gradient
                        for layer=1:obj.nbrofLayers
                            % vectorize the layer weights and bias by
                            % reshaping first two dimensions into a vector
                            DF=reshape(DeltaF{layer},prod(size(DeltaF{layer},[1 2])),nbrofOutUnits*nbrofSamplesinBatch);
                                                        
                            % assign the appropriate portion of the
                            % jacobian so it's easier to keep track
                            Jac([obj.layerweightinds{layer} obj.layerbiasinds{layer}],:) = DF;       
                            
                        end                       
                        Jac = Jac / sqrt(nbrofSamplesinBatch);
                        Hessian = (Jac*Jac');
                        
                        % Only use this line for real only case:
                        % jace = Jac*transpose(curr_normalized - trn_normalized);
                        % this line works for real and complex case:
                        jace = Weights_to_vec(obj,DeltaW);                        
                        
                        %--------------------------------------------------
                        %        D E B U G with MATLAB jacobian
                        % this is mostly code for deep numerical dive
                        % use comparenets.m to set this up
                        %--------------------------------------------------                        
                        if obj.debugCompare
                            debug_lm(obj,Jac,jace,Hessian,matlablayerweightinds,matlablayerbiasinds);
                        end               
                        
                        trn = out(:,batchtrain);
                        msetrn = inf; numstep = 1;
                        ee = eye(size(Hessian));

                        % generally, QR decomposition has better behavior
                        % than computing Hessian via Jac*Jac'
                        USEQR = 1; 
                        % fast update is rank one updates to base qr(Jac')
                        FAST_UPDATE = 0;
                        if USEQR && FAST_UPDATE,  R = qr(Jac'); end
                        while (msetrn>msecurr) && (obj.mu<obj.mu_max)                            
                            % Levenberg Marquardt                            
                            Hblend = Hessian + obj.mu*ee;
                            %Hblend = Hessian + (obj.mu*diag(diag(Hessian)));
                            
                            if isnan(rcond(Hblend))
                                error('Condition number of blended Hessian is nan.  What did you do?');
                            end                            
                            
                            %----------------------------------------------
                            % Blended Newton / Gradient 
                            % (i.e. LevenBerg-Marquardt)                            
                            if USEQR==0
                                WbWb = Hblend\jace;
                            else
                                % use qr decomposition instead of Hessian which
                                % is obtained by squaring
                                if FAST_UPDATE
                                    R1 = triu(qrupdate_al( R, sqrt(obj.mu)));
                                else
                                    Jact = [Jac sqrt(obj.mu)*eye(obj.totalnbrofParameters)]';
                                    R1 = triu(qr(Jact,0));
                                end                                
                                WbWb = R1\(R1'\jace);                                                                                                
                                % find extra step e based on residual error r
                                % does not generally help
                                %r = jace - Hblend*WbWb;
                                %e = R1\(R1'\r);                               
                            end
                            %----------------------------------------------                            
                                                        
                            % convert vector to Weights
                            DeltaW = vec_to_Weights(obj,WbWb);
                            for layer=1:obj.nbrofLayers                              
                                if obj.debugPlots && printthisepoch
                                    figure(1024);
                                    subplot(obj.nbrofLayers,1,layer)
                                    imagesc(real(DeltaW{layer})); colorbar;
                                    figure(101); imagesc(real(Hessian)); colorbar;
                                end                                
                                obj.Weights{layer} = obj.Weights{layer} -1*DeltaW{layer};
                            end
                            
                            curr = obj.test( in(:,batchtrain) );                            
                            % Check the mse for the update
                            msetrn = mean( abs(curr(:)-trn(:)).^2 );
                            if printthisepoch
                                fprintf('mu %e msetrn %e\n',obj.mu,msetrn);
                            end
                            
                            %{
                            % trying out the residual step, generally
                            % doesn't help
                            ErrorW = vec_to_Weights(obj,e);
                            for nn=1:obj.nbrofLayers                              
                                obj.Weights{nn} = obj.Weights{nn} - ErrorW{nn};
                            end
                            curre = obj.test( in(:,batchtrain) );
                            % Check the mse for the update
                            msetrne = mean( abs(curre(:)-trn(:)).^2 );
                            fprintf('mu %e msetrn(with error step) %e\n',mu,msetrne);
                            ErrorW = vec_to_Weights(obj,e);
                            for nn=1:obj.nbrofLayers                              
                                obj.Weights{nn} = obj.Weights{nn} + ErrorW{nn};
                            end
                            %}                            
                            
                            if isnan(msetrn)
                                any(isnan(obj.Weights{1}))
                                any(isnan(objnew.Weights{1}))
                            end
                            numstep=numstep+1;                            

                            if msetrn>msecurr
                                obj.mu=obj.mu*obj.mu_inc; % pad the Hessian more
                                % undo the weight step
                                for layer=1:obj.nbrofLayers
                                    obj.Weights{layer} = obj.Weights{layer} +1*DeltaW{layer};
                                end
                            else
                                % as mu decreases, becomes Newton's method
                                obj.mu = max(obj.mu * obj.mu_dec,obj.mu_min);
                            end
                        end                                                
                       
                    otherwise                        
                        %--------------------------------------------------
                        % simple gradient based approaches are easy to
                        % update

                        thegradientfunction = str2func(obj.trainFcn);
                        for layer=1:obj.nbrofLayers
                            try
                                [DeltaW{layer},state{layer}] = ...
                                    thegradientfunction(DeltaW{layer},state{layer});
                            catch
                                error('Unable to train using %s. Path must have ./gradient_descent',obj.trainFcn);
                            end
                            obj.Weights{layer} = obj.Weights{layer} - DeltaW{layer};
                        end
                end
                
                % if you want a history of the weights in vector form
                %obj.WeightsHistory(:,epoch) = Weights_to_vec(obj,Weights,layerweightinds,layerbiasinds,totalnbrofParameters);
                WbWb = Weights_to_vec(obj,obj.Weights);
                obj.WeightsCircularBuffer(:,1:end-1) = obj.WeightsCircularBuffer(:,2:end);
                obj.WeightsCircularBuffer(:,end) = WbWb;
                
                
                % Check the mses for the update
                curr = obj.test( in(:,batchtrain) );
                msetrn = mean( abs(curr(:)-trn(:)).^2 );            
                curr = obj.test( in(:,batchtest) );
                msetst = mean( abs(curr(:)-tst(:)).^2 );                                
                curr = obj.test( in(:,batchvalidate) );
                msevl = mean( abs(curr(:)-vl(:)).^2 );                                
                                             
                msetrain(epoch)= msetrn;
                msetest(epoch) = msetst;
                msevalidate(epoch) = msevl;

                %if (epoch>max_fail)
                %    msemin = min(msetest(1: epoch-max_fail) );
                %    if msetst > msemin + eps
                %        testfail = max_fail;
                %    end
                %end
                
                if epoch>1 
                    if msevl > msevalidate(epoch-1) + eps
                        testfail = testfail + 1;   % count fail to decrease
                    else
                        % reset to zero, allowing for up down fluctuation
                        % primarily at the beginning of training
                        testfail = 0;              
                    end
                end          
                
                epochtime = toc(tstart);
                                
                if printthisepoch
                    fprintf('epoch %d: msetrain %f msetest %f msevalidate %f grad %e\n',...
                        epoch,(msetrn),(msetst),(msevl),gradientacc);
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
            
            
            if epoch>obj.max_fail
                if kt.fail==0
                    ind = 1;
                else
                    [msevl,ind] = min( msevalidate(epoch-obj.max_fail:epoch) );
                end
                fprintf('Scrolling back weights by %d epochs to best at epoch %d\n',...
                    (obj.max_fail - ind + 1), epoch - obj.max_fail + ind-1);
                WbWb = obj.WeightsCircularBuffer(:,ind);  %max_fail + 1 buffer
                obj.Weights = vec_to_Weights(obj,WbWb);
                msetrn = msetrain(epoch-obj.max_fail + ind-1);
                msetst = msetest(epoch-obj.max_fail + ind-1);
                msevl = msevalidate(epoch-obj.max_fail + ind-1);
            end
            
            fprintf('time to train %0.3f sec, exit epoch %d: msetrain %f msetest %f msevalidate %f\n\n',...
                epochtime,epoch,(msetrn),(msetst),(msevl));
            fprintf('keeptraining flags (0 = exit condition reached):\n');
            disp(kt);
            
            if obj.performancePlots
                figure(231); clf;
                semilogy(msetrain(1:epoch),'b.-','MarkerSize',20); hold on;
                semilogy(msetest(1:epoch),'r+-','MarkerSize',4);
                semilogy(msevalidate(1:epoch),'go-','MarkerSize',4);
                semilogy(1:epoch,msevl*ones(1,epoch),'.','MarkerSize',4)
                legend('train (for determining weights)',...
                    'test (for reporting performance)',...
                    'validate (exit on 6 consequetive increases)',...
                    'optimal weights','FontSize',18,'FontWeight','bold'); 
                grid minor;
                xlabel('epoch','FontSize',18,'FontWeight','bold');
                ylabel('mean-squared error','FontSize',18,'FontWeight','bold');
                title('Trajectory of performance','FontSize',18,'FontWeight','bold');
            end            
            
        end
    end
end

