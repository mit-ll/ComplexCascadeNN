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

        trainFcn     % see options in gradient_descent directory
                     % Hessian-based 'trainlm'=Levenberg-Marquardt
                     
        mu_inc
        mu_dec
        mu_max
        mu_min
                     
        initFcn      % 'randn','nguyen-widrow'

        minbatchsize
        batchtype    %'randperm','fixed'
        
        nbrofEpochs
        
        % mapminmax
        domap
        inputSettings
        outputSettings
        
        % for debugging
        debugPlots
        
        % for comparison with MATLAB feedforwardnet output
        debugCompare
        weightRecord
        jeRecord
        jjRecord        
        weightRecord2
        
    end
    
    properties (Constant)
        nbrofEpochsdefault = 1e3; % number of iterations picking a batch each time and running gradient
        printmseinEpochs = 1;  % one print per printmeseinEpochs
        beta1=0.9;         % Beta1 is the decay rate for the first moment
        beta2=0.999;       % Beta 2 is the decay rate for the second moment
        
        lrate_default = 1e-2; % initial step size for the gradient        

        initial_mu = 1e-3; % Hessian + mu * eye for Levenberg-Marquardt
               
        batchsize_per_feature = 50;    % number of samples per feature to use at a time in a epoch
        minbatchsizedefault = 25;
        batchtypedefault = 'randperm';
        
        epochs_drop = 100;   % number of epochs before dropping learning rate
        drop = 1/10;         % drop in learning rate new = old*drop
        

        % some stopping parameters
        maxtime = 60*100;    % (s) compared to tic/toc
        msedesired = 0;
        min_grad = 1e-7;
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
            [dz.real,dz.imag] = deal(zeros(size(z)));
            dz.real(~indsr) = 1;
            dz.imag(~indsi) = 1;                        
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
            if 0
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
            
            % map to [-1 1]
            if isfield(params,'domap') 
                obj.domap=params.domap; 
            else
                obj.domap = 0;
            end

            
            if isfield(params,'debugPlots')
                obj.debugPlots=params.debugPlots;
            else
                obj.debugPlots=0;
            end                                                
                                                
            % hidden layers sizes is a vector 
            obj.hiddenSize = params.hiddenSize;
            obj.layers = cell(1,length(obj.hiddenSize)+1);

            outputFcn = params.outputFcn;
            layersFcn = params.layersFcn;
            
            % hidden layers
            for ll=1:length(obj.layers)-1
                if iscell(layersFcn)
                    if numel(layersFcn)==length(obj.layers)
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
                
                if 1 || (epoch/obj.printmseinEpochs == floor(epoch/obj.printmseinEpochs))
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
        function [out,x,y0,y,yprime] = test(obj,in)
                        
            if obj.domap
                % scale the input
                ins = mapminmax.apply(obj.realify(in),obj.inputSettings);
                in = obj.unrealify(ins);
            else
                % gain only
                %in = bsxfun(@times,in,obj.inputSettings.gain);
            end
            
            % include bias vector for first layer
            % bias can be real since weights are complex
            biasvec = ones(1,size(in,2));
            y0 = [in; biasvec];

            % used for derivative, but unnecessary since the corresponding
            % column is eliminated in the derivation of dc/dx
            zerovec = zeros(1,size(in,2));              
            
            ynnminus1 = y0;                        
            [x,y,yprime] = deal(cell(1,obj.nbrofLayers));
            for nn=1:obj.nbrofLayers
                % weight matrix
                W=obj.Weights{nn};
                
                % transfer function
                transferFcn = obj.layers{nn}.transferFcn;  % string
                activationfn = @(x) obj.(transferFcn)(x);  % handle
                
                % apply matrix of weights
                x{nn} = W*ynnminus1;     % x{n} = W{n}y{n-1}
                
                % evaluate f(xn) and f'(xn)
                [y{nn},yprime{nn}] = activationfn(x{nn});
                
                % include a bias vector for next layer
                % final out y{nbrofLayers} does not have bias
                if nn < obj.nbrofLayers
                    % do not include the derivative of the bias w.r.t x{n}
                    % which is the zerovec
                    %yprime{nn} = [yprime{nn}; zerovec];
                    
                    % do include (additional row) the biasvec as part of y
                    y{nn} = [y{nn}; biasvec];
                end
                ynnminus1 = y{nn};
            end
               
            out_normalized = y{obj.nbrofLayers};
            if obj.domap
                % network matches to -1,1 but output is back to original scale
                out = mapminmax.reverse( obj.realify(out_normalized),obj.outputSettings);      
                out = obj.unrealify(out);                
            else
                % gain only
                %out = bsxfun(@rdivide,out_normalized,obj.outputSettings.gain);
                out=out_normalized;
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
        
        %------------------------------------------------------------------
        function obj = train(obj,in,out)
            % in  is features x number of training samples
            % out is desired output x number of training samples            
            
            if obj.domap
                % per feature normalization into -1 1
                % y = (ymax-ymin)*(x-xmin)/(xmax-xmin) + ymin
                % y = ymin + gain * (x - xmin)
                % dx_dy = 1/gain
                [in_normalized,obj.inputSettings] = mapminmax(obj.realify(in));

                %warning('setting real and imag input gain to be same');
                %gg = obj.inputSettings.gain;
                %gg = max([gg(1:end/2) gg(end/2+1:end)],[],2);
                %gg = repmat(gg,2,1);
                %obj.inputSettings.gain = gg;                
                
                [out_normalized,obj.outputSettings] = mapminmax(obj.realify(out));
                
                %warning('setting real and imag output gain to be same');
                %gg = obj.outputSettings.gain;
                %gg = max([gg(1:end/2) gg(end/2+1:end)],[],2);
                %gg = repmat(gg,2,1);
                %obj.outputSettings.gain = gg;                           
                
                in_normalized = obj.unrealify(in_normalized);
                out_normalized = obj.unrealify(out_normalized);                
                                
                derivative_outputmap = mapminmax('dx_dy',obj.realify(out(:,1)),obj.realify(out_normalized(:,1)),obj.outputSettings);
                derivative_outputmap = cell2mat(derivative_outputmap);
                
            else
                % use gain only map
                %obj.inputSettings.gain = 1./max(abs(in),[],2);
                %obj.outputSettings.gain = 1./max(abs(out),[],2);
                %in_normalized = bsxfun(@times,in,obj.inputSettings.gain);
                %out_normalized = bsxfun(@times,out,obj.outputSettings.gain);
                
                % no normalization for real or imag
                obj.inputSettings.gain = ones(2*size(in,1),1);
                obj.outputSettings.gain = ones(2*size(out,1),1);                
                in_normalized=in;
                out_normalized=out;
                
                derivative_outputmap = diag(1./obj.outputSettings.gain);
            end
            
            [nbrofInUnits, nbrofSamples] = size(in);
            [nbrofOutUnits, nbr] = size(out);
            if nbr~=nbrofSamples
                error('input and output number of samples must be identical');
            end            
            nbrOfNeuronsInEachHiddenLayer = obj.hiddenSize;
             
            % pick batch size number of sample if possible
            nbrofSamplesinBatch =  max(obj.batchsize_per_feature*nbrofInUnits,obj.minbatchsize);
            nbrofSamplesinBatch =  min(nbrofSamplesinBatch,nbrofSamples);            
      
            % allocate space for gradients
            % deltax{n} = dcost/dx{n} vector of dimension x{n}
            % DeltaW{n} = dcost/dWeights{n} matrix of dimension W{n}
            % state is cell array with state of gradient and hessian approx
            % there is redundancy here since each layer has it's own state
            % that has rate parameters that are same - not a big deal
            [deltax,DeltaW,state] = deal(cell(1,obj.nbrofLayers));

            switch obj.trainFcn                        
                case 'trainlm'            
                    [Deltaf] = deal(cell(1,obj.nbrofLayers));
                    mu = obj.initial_mu;
                otherwise
                    % need this for keeptraining check even though unused
                    % for other trainFcn's
                    mu = obj.initial_mu;
            end
            
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

            for nn=1:obj.nbrofLayers                    
                % warning: W*x used here instead of W'*x
                % since y = W*x, the dimensions of W are out x in 
                % NOT in x out                                
                switch obj.initFcn
                    case 'crandn'
                        obj.Weights{nn} = 0.01*obj.crandn( obj.nbrofUnits(2,nn), obj.nbrofUnits(1,nn) );                        
                    case 'c-nguyen-widrow'
                        %Matlab's Nguyen-Widrow Algorithm
                        obj.Weights{nn} = zeros(  obj.nbrofUnits(2,nn), obj.nbrofUnits(1,nn) );
                        activeregionofactivation = [-2 2];  % tansig('active');
                        mapregion = [-1 1];
                        nbrofNeurons = obj.nbrofUnits(2,nn);
                        nbrofIn = obj.nbrofUnits(1,nn)-1;
                        mapregions = repmat(mapregion,nbrofIn,1);
                        [wr,br]=mycalcnw(mapregions,nbrofNeurons,activeregionofactivation);
                        [wi,bi]=mycalcnw(mapregions,nbrofNeurons,activeregionofactivation);                        
                        obj.Weights{nn} = [wr+1i*wi br+1i*bi];                        
                        
                    %-----------------------------
                    % non-complex initializations                                        
                    case 'randn'
                        obj.Weights{nn} = 0.01*randn( obj.nbrofUnits(2,nn), obj.nbrofUnits(1,nn) );                        

                    case 'nguyen-widrow'
                        %Matlab's Nguyen-Widrow Algorithm
                        obj.Weights{nn} = zeros(  obj.nbrofUnits(2,nn), obj.nbrofUnits(1,nn) );
                        activeregionofactivation = [-2 2];  % tansig('active');
                        mapregion = [-1 1];
                        nbrofNeurons = obj.nbrofUnits(2,nn);
                        nbrofIn = obj.nbrofUnits(1,nn)-1;
                        mapregions = repmat(mapregion,nbrofIn,1);
                        [w,b]=mycalcnw(mapregions,nbrofNeurons,activeregionofactivation);
                        obj.Weights{nn} = [w b];
                        
                    % weights already initialized                        
                    case 'previous'
                        if isempty(obj.Weights)
                            error('change obj.initFcn=%s, since previous weights are empty',obj.initFcn);
                        end
                        if any( size(obj.Weights{nn})~= [obj.nbrofUnits(2,nn), obj.nbrofUnits(1,nn)])
                            fprintf('layer%d Weights %d x %d ~= dimensions %d x %d\n',nn,...
                                size(obj.Weights{nn},1),size(obj.Weights{nn},2),...
                                obj.nbrofUnits(2,nn), obj.nbrofUnits(1,nn));
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
                switch obj.initFcn
                    case 'previous'
                    otherwise
                        fprintf('setting %d bias weights to zero for layer %d to initialize\n',obj.nbrofUnits(2,nn),nn);
                        obj.Weights{nn}(:,end)=0;
                end
                %}
                
                [DeltaW{nn}] = ...
                    deal(zeros( obj.nbrofUnits(2,nn), obj.nbrofUnits(1,nn) ));   
                
                switch obj.trainFcn            
                    case 'trainlm'            
                        % Deltaf matrices are due to weights only 
                        if nn==obj.nbrofLayers
                            Deltaf{nn} = zeros(nbrofOutUnits,nbrofOutUnits,nbrofSamplesinBatch);
                        else
                            Deltaf{nn} = zeros(obj.nbrofUnits(2,nn),nbrofOutUnits,nbrofSamplesinBatch);                            
                        end
                end
            end
            
            obj.totalnbrofParameters = sum(obj.nbrofWeights+obj.nbrofBias); 
            switch obj.trainFcn
                case 'trainlm'
                    Jac = zeros(obj.totalnbrofParameters,nbrofSamplesinBatch);
                    % Hessian and Jac*error are derived from Jac and don't
                    % need to be pre allocated
                    %Hessian = zeros(totalnbrofParameters,totalnbrofParameters);
                    %jace = zeros(totalnbrofParameters,1);                                        
                    [deltaf,DeltaF] = deal(cell(1,obj.nbrofLayers));                    
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
            
            % update the weights over the epochs
            [msetrain,msetest,msevalidate] = deal(-1*ones(1,obj.nbrofEpochs));
            lrate = obj.lrate;
            disp('learning rate is not being used');     
            
            epoch=0; keeptraining = 1; 
            testfail = 0;
            tstart = tic;
            while keeptraining
                epoch=epoch+1;
                
                printthisepoch = (epoch/obj.printmseinEpochs == floor(epoch/obj.printmseinEpochs));
                
                % pick a batch for this epoch for training
                switch obj.batchtype
                    case 'randperm'
                        batchtrain = randperm(nbrofSamples,nbrofSamplesinBatch);
                    case 'fixed'
                        batchtrain = 1:nbrofSamplesinBatch;
                end                
                % testing and validation each get 1/2 of the remainder
                batchleft = setdiff(1:nbrofSamples,batchtrain); L = length(batchleft);
                L2 = floor(L/2);                
                batchtest = batchleft(1:L2);                
                batchvalidate = batchleft(L2+1:end);
                
                % training, testing, and validation
                trn_normalized = out_normalized(:,batchtrain);
                trn = out(:,batchtrain);                
                tst_normalized = out_normalized(:,batchtest);
                tst = out(:,batchtest);                
                vl_normalized = out_normalized(:,batchvalidate);
                vl = out(:,batchvalidate);
                
                % evaluate network and obtain gradients and intermediate values
                % y{end} is the output
                [curr,x,y0,y,yprime] = obj.test( in(:,batchtrain) );
                curr_normalized = y{obj.nbrofLayers};
                
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
                % backpropagate to get all layers                
                splitrealimag = zeros(1,obj.nbrofLayers);
                largestgradient = 0;
                for nn=obj.nbrofLayers:-1:1                             
                    % deltax is the gradient at the middle of the layer
                    % deltax = d{cost}/d{xn} called "sensitivity"
                    % 
                    deltax{nn} = zeros( obj.nbrofUnits(2,nn), nbrofSamplesinBatch);
                    
                    if isstruct(yprime{nn}), splitrealimag(nn) = 1; else, splitrealimag(nn)=0; end
                    
                    if nn==obj.nbrofLayers
                        % for output layer, using mean-squared error
                        % w = w - mu (y-d) f'(net*) x*                        

                        % have to include the dx/dy = 1/gain on the output
                        % since mse is computed there
                        % @todo: a better way is to make a constant output weighting node
                        err_times_outputmap_derivative = obj.unrealify(derivative_outputmap * obj.realify(curr-trn));
                        
                        if splitrealimag(nn)                            
                            deltax{nn} = real(err_times_outputmap_derivative) .* yprime{nn}.real + ...
                                1i* imag(err_times_outputmap_derivative).* yprime{nn}.imag;                                                        
                            deltaf{nn} = yprime{nn}.real + 1i* yprime{nn}.imag;                            
                        else                            
                            deltax{nn} = err_times_outputmap_derivative .* conj( yprime{nn} );                            
                            deltaf{nn} = conj( yprime{nn} );         
                            
                            % trying out split re/im even though it's not
                            % correct
                            %
                            %deltax{nn} = real(err_times_outputmap_derivative) .* real( yprime{nn} ) + ...
                            %    1i* imag(err_times_outputmap_derivative).* imag( yprime{nn} );                                                        
                            %deltaf{nn} = real( yprime{nn} ) + 1i* imag( yprime{nn} );                            
                        end                        
                    else                         
                        dx_nnplus1 = deltax{nn+1}; % =0, assigned just for dimensions
                        yp_nn = yprime{nn};
                        W_nnplus1 = obj.Weights{nn+1};
                        df_nnplus1 = deltaf{nn+1}; % =0, assigned just for dimensions
                        
                        if splitrealimag(nn)
                            % for 2-layer derivation, see equation(17)
                            % "Extension of the BackPropagation
                            % algorithm to complex numbers"
                            % by Tohru Nitta
                            ypr = yp_nn.real;
                            ypi = yp_nn.imag;
                            dxr_nnplus1 = real(dx_nnplus1);
                            dxi_nnplus1 = imag(dx_nnplus1);
                            deltax{nn} = ypr.* ...
                                (  (real(W_nnplus1(:,1:end-1)).' * dxr_nnplus1) +...
                                (imag(W_nnplus1(:,1:end-1)).' * dxi_nnplus1) ) +...
                                -1i* ypi .* ...
                                (  (imag(W_nnplus1(:,1:end-1)).' * dxr_nnplus1) +...
                                   -1*(real(W_nnplus1(:,1:end-1)).' * dxi_nnplus1 ) );
                            
                            dfr_nnplus1 = real(df_nnplus1);
                            dfi_nnplus1 = imag(df_nnplus1);                            
                            deltaf{nn} = ypr.* ...
                                (  (real(W_nnplus1(:,1:end-1)).' * dfr_nnplus1) +...
                                (imag(W_nnplus1(:,1:end-1)).' * dfi_nnplus1) ) +...
                                -1i* ypi .* ...
                                (  (imag(W_nnplus1(:,1:end-1)).' * dfr_nnplus1) +...
                                   -1*(real(W_nnplus1(:,1:end-1)).' * dfi_nnplus1 ) );
                        else
                            % last column of Weights are from the bias
                            % term for nn+1 layer, which does not
                            % contribute to nn layer, hence 1:end-1
                            deltax{nn} = conj(yp_nn).* ( W_nnplus1(:,1:end-1).' * dx_nnplus1);
                            deltaf{nn} = conj(yp_nn).* ( W_nnplus1(:,1:end-1).' * df_nnplus1);
                        end
                    end % if nn=nbrofLayers, i.e. last layer
                    
                    if nn==1 
                        ynnminus1 = y0; % no previous layer for first layer
                    else
                        ynnminus1 = y{nn-1}; 
                    end
                    
                    % obj.nbrofUnits(2,nn) x obj.nbrofUnits(1,nn)
                    % outer product of deltax with ynnminus1, summed over
                    % the measurements                    
                    DeltaW{nn} = deltax{nn}*ynnminus1'/nbrofSamplesinBatch; 

                    for ll = 1:obj.nbrofUnits(2,nn)    
                        dF = zeros(obj.nbrofUnits(1,nn),nbrofSamplesinBatch);
                        dF(1:end-1,:) = bsxfun(@times,conj(ynnminus1(1:end-1,:)),deltaf{nn}(ll,:));       
                        dF(end,:) = deltaf{nn}(ll,:);
                        DeltaF{nn}(ll,:,:) = dF;
                    end    
                    
                    if obj.debugPlots && printthisepoch
                        figure(1023);
                        subplot(obj.nbrofLayers,1,nn)
                        imagesc(real(DeltaW{nn})); colorbar;
                    end
                    largestgradient = max(largestgradient, max(abs(DeltaW{nn}(:))));
                end
                  
                %for linear single neuron 
                % DeltaW = (y{nn} - t) .* conj( yprime{nn} ) * ynnminus1'/nbrofSamplesinBatch;      
                % Jac = conj( yprime{nn} ) * conj(ynnminus1(1:end-1,mm)));
                % jace += transpose(Jac) * (y{obj.nbrofLayers}(:,mm) - t(:,mm));
                
                % update all the weight matrices (including bias weights)
                %
                switch obj.trainFcn
                    case 'trainlm'
                        % compute jacobian matrix
                        for nn=1:obj.nbrofLayers
                            % vectorize the layer weights and bias by
                            % reshaping first two dimensions into a vector
                            DF=reshape(DeltaF{nn},prod(size(DeltaF{nn},[1 2])),nbrofSamplesinBatch);
                            
                            % assingn the appropriate portion of the
                            % jacobian so it's easier to keep track
                            Jac([obj.layerweightinds{nn} obj.layerbiasinds{nn}],:) = DF;
                            
                        end                       
                        Jac = Jac / (obj.outputSettings.gain(1) * sqrt(nbrofSamplesinBatch));
                        Hessian = (Jac*Jac');
                        
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
                        jace = Weights_to_vec(obj,DeltaW);                                                
                        
                        
                        %--------------------------------------------------
                        %        D E B U G with MATLAB jacobian
                        % this is mostly code for deep numerical dive
                        %--------------------------------------------------                        
                        if obj.debugCompare
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
                                                                
                                somescalar = mean(jace./jertocompare);                                
                                somescalar2 = mean(Hessian(:)./jjrtocompare(:))
                                                                
                                norm(jace/somescalar - jertocompare)                                
                                norm(Hessian/somescalar2 - jjrtocompare)
                                                                                                
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
                                %norm(dW1lsq - dW2)                                  
                                dW1lsq = lsqminnorm(Jact'*Jact,jace);
                                
                                % svd approach
                                [~,S,V] = svd(Jact);
                                svec = diag(S); svec2 = 1./(svec.*conj(svec));
                                dW1svd = (V*diag(svec2)*V')*jace;                                
                                
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
                                
                                disp('d(cnet svd - matlab)');                                      
                                norm(dW1svd - dWmatlab)
                                
                            catch
                                Jac
                                Hessian
                            end
                            keyboard;
                        end
                        %--------------------------------------------------
                        %        D E B U G
                        %--------------------------------------------------
                        
                        trn = out(:,batchtrain);
                        msetrn = inf; numstep = 1;
                        ee = eye(size(Hessian));
                        while (msetrn>msecurr) && (mu<obj.mu_max)                            
                            % Levenberg Marquardt                            
                            %Hblend = Hessian + mu*ee*(1  + 1i*any(imag(Hessian(:))));
                            Hblend = Hessian + mu*ee;
                            %Hblend = Hessian + (mu*diag(diag(Hessian)));
                            %Hblend = Hessian + mu*ee*( max(real(diag(Hessian))) + 1i*max(imag(diag(Hessian))));
                            
                            if isnan(rcond(Hblend))
                                error('Condition number of blended Hessian is nan.  What did you do?');
                            end                            
                            
                            %----------------------------------------------
                            % Blended Newton / Gradient 
                            % (i.e. LevenBerg-Marquardt)                            
                            if 0
                                WbWb = Hblend\jace;
                            else
                                % use qr decomposition instead of Hessian which
                                % is obtained by squaring
                                Jact = [Jac sqrt(mu)*eye(obj.totalnbrofParameters)]';
                                R1 = triu(qr(Jact));
                                WbWb = R1\(R1'\jace);
                                
                                % find extra step e based on residual error r
                                A = (Jact'*Jact);
                                r = jace - A*WbWb;
                                e = R1\(R1'\r);
                                WbWb = WbWb + e;  
                            end
                            %----------------------------------------------                            
                                                        
                            % convert vector to Weights
                            DeltaW = vec_to_Weights(obj,WbWb);
                            for nn=1:obj.nbrofLayers                              
                                if obj.debugPlots && printthisepoch
                                    figure(1024);
                                    subplot(obj.nbrofLayers,1,nn)
                                    imagesc(real(DeltaW{nn})); colorbar;
                                    figure(101); imagesc((Hessian)); colorbar;
                                end                                
                                obj.Weights{nn} = obj.Weights{nn} -  DeltaW{nn};
                            end
                            
                            curr = obj.test( in(:,batchtrain) );
                            % Check the mse for the update
                            msetrn = mean( abs(curr(:)-trn(:)).^2 );
                            fprintf('mu %e msetrn %f\n',mu,msetrn);
                            if isnan(msetrn)
                                any(isnan(obj.Weights{1}))
                                any(isnan(objnew.Weights{1}))
                            end
                            numstep=numstep+1;                            

                            if msetrn>msecurr
                                mu=mu*obj.mu_inc; % pad the Hessian more
                                % undo the weight step
                                for nn=1:obj.nbrofLayers
                                    obj.Weights{nn} = obj.Weights{nn} +  DeltaW{nn};
                                end
                            else
                                % as mu decreases, becomes Newton's method
                                mu = max(mu * obj.mu_dec,obj.mu_min);
                            end
                        end                                                
                       
                        %{
                        % switch to slow gradient at the end
                        if mu>=obj.mu_max
                            newtrainFcn = 'Adadelta';
                            warning('Switching from %s to %s\n',obj.trainFcn);
                            obj.trainFcn = newtrainFcn;
                            mu=0;
                        end 
                        %}

                        
                    otherwise
                        thegradientfunction = str2func(obj.trainFcn);
                        for nn=1:obj.nbrofLayers
                            try
                                [DeltaW{nn},state{nn}] = ...
                                    thegradientfunction(DeltaW{nn},state{nn});
                            catch
                                error('Unable to train using %s. Path must have ./gradient_descent',obj.trainFcn);
                            end
                            obj.Weights{nn} = obj.Weights{nn} - DeltaW{nn};
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
                    fprintf('epoch %d: msetrain %f msetest %f msevalidate %f\n',...
                        epoch,(msetrn),(msetst),(msevl));
                end                                
                
                kt.epoch = (epoch < obj.nbrofEpochs);
                kt.time = (epochtime < obj.maxtime);
                kt.mu = (mu < obj.mu_max);
                kt.mse = (msetrn > obj.msedesired);
                kt.grad = (largestgradient > obj.min_grad);
                kt.fail = (testfail < obj.max_fail);
                keeptraining = all( struct2array(kt) );
                
            end % while keeptraining
            
            if kt.fail==0
                fprintf('Scrolling back weights by %d epochs to best at epoch %d\n',...
                    obj.max_fail,epoch-obj.max_fail); 
                WbWb = obj.WeightsCircularBuffer(:,1);
                obj.Weights = vec_to_Weights(obj,WbWb);
                msetrn = msetrain(epoch-obj.max_fail);
                msetst = msetest(epoch-obj.max_fail);
                msevl = msevalidate(epoch-obj.max_fail);
            end
            
            fprintf('time to train %0.3f sec, exit epoch %d: msetrain %f msetest %f msevalidate %f\n\n',...
                epochtime,epoch,(msetrn),(msetst),(msevl));
            fprintf('keeptraining flags (0 = exit condition reached):\n');
            disp(kt);
            
            if obj.debugPlots || 1
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

