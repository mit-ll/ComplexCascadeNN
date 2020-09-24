classdef complexnet < handle
    % complexnet Feed-forward neural network with variety of
    % backpropagation approaches
    %
    % Swaroop Appadwedula Gr.64 
    % August 13,2020 in the time of COVID
    
 
    %{
    %----------------------------------------------------------------------
    % check for single complex layer    
    params.hiddenSize=[]; params.outputFcn='purelin';
    params.trainFcn='trainlm';%'grad';
    params.initFcn='c-nguyen-widrow';
    net = complexnet(params)
    x=randn(1,100)+1i*randn(1,100); 
    net.trainsingle(x,x*(3+5*1i) + (4-2*1i));
    net = net.train(x,x*(3+5*1i) + (4-2*1i)); 
 
    x=randn(1,100); 
    net.initFcn='nguyen-widrow';
    net = net.train(x,3*x + 2)    
    print(net);
    
    % phase
    params.hiddenSize=[]; params.outputFcn='purephase'; net = complexnet(params)
    x=randn(1,100)+1i*randn(1,100); 
    net.trainsingle(x,x./abs(x));    
    net.train(x,(3+3*i)* x./abs(x));  % does not work - can't handle scalar
    print(net);
    
    % sigmoid
    params.hiddenSize=[]; params.outputFcn='sigrealimag'; net = complexnet(params)
    x=randn(1,10000)+1i*randn(1,10000); 
    y = net.sigrealimag(x);
    [W,out] = net.trainsingle(x,y);    
    plot(real(y),real(out),'.'); hold on;
    plot(imag(y),imag(out),'o');
    
    % input 2 x output 10 checking dimensions of w and Levenberg-Marquardt
    params.hiddenSize=[]; params.outputFcn='purelin'; 
    params.batchtype = 'randperm';
    params.minbatchsize=10;
    net = complexnet(params)
    x = randn(2,100)+1i*randn(2,100); 
    w = (randn(10,2)+1i*randn(10,2));
    b = (randn(10,1)+1i*randn(10,1));
    net.trainsingle(x, bsxfun(@plus,w * x,b));
    net.Weights{1}
    w
    net.Weights{2}
    b
    
    %----------------------------------------------------------------------    
    % check for multiple real layers
    params.hiddenSize=[1 2 2]; params.outputFcn='purelin'; net = complexnet(params)
    x=randn(1,200); net = net.train(x,[x; 3*x.^2 + 2*ones(size(x)); 10*x]);
    print(net)
    
    % check for multiple complex layers
    params.hiddenSize=[1 2 2]; params.outputFcn='purelin'; net = complexnet(params)
    x=randn(1,200)+1i*randn(1,200); net = net.train(x,x*(3+5*1i) + (4-2*1i))        
    print(net)
    
    params.hiddenSize=[1]; params.layersFcn='purephase'; params.outputFcn='purelin'; net = complexnet(params)    
    x=randn(1,200)+1i*randn(1,200); net = net.train(x,(3+3*i)* (x./abs(x)) );
    print(net)            
    
    %----------------------------------------------------------------------
    % rotation example (as in Nitta)
    % ellipse
    % I
    rot = pi/3; R = [cos(rot) -sin(rot); sin(rot) cos(rot)];  
    %R = diag([0.5 0.3])*R;
    
    ini1 = linspace(-0.9,0.9,7); inr1 = zeros(size(ini1));
    inr2 = linspace(-0.2,0.2,2); ini2 = 0.95*ones(size(inr2));
    inr3 = inr2; ini3=-ini2;    
    shapeI=[inr1 inr2 inr3] + 1i*[ini1 ini2 ini3];
        
    A = 0.6; B = 0.3; th = linspace(0,2*pi,220);
    shapeO = A*cos(th) + 1i*B*sin(th);

    shape = shapeI;    
    y= R * [real(shape); imag(shape)];
    shaperotated = y(1,:) + 1i*y(2,:);
    
    y= R * [real(shapeO); imag(shapeO)];
    shapeOrotated = y(1,:) + 1i*y(2,:);
    
    params.initial_lrate=1e-2;
    params.nbrofEpochs=1e3;
    params.hiddenSize=[1 2 1]; 
    params.layersFcn='sigrealimag2'; params.outputFcn='purelin'; 
    params.initFcn='c-nguyen-widrow';
    params.minbatchsize=inf;params.batchtype = 'fixed';
    params.trainFcn ='trainlm'; %'adam';    
        
    cnet = complexnet(params)    
    cnet = cnet.train(shape,shaperotated);
    outhat = cnet.test(shape);
    outO = cnet.test(shapeO);
    print(cnet)        

    net = feedforwardnet( 2* params.hiddenSize);
    realifyfn = @(x) [real(x); imag(x)];
    net = train(net,realifyfn(shape),realifyfn(shaperotated));
    outOr =net(realifyfn(shapeO)); outOr=outOr(1,:)+1i*outOr(2,:); 
    
    figure(123);clf;
    plot(shape,'.','MarkerSize',12); hold on;
    plot(shaperotated,'v','MarkerSize',12);
    plot(outhat,'s','MarkerSize',12);
    
    plot(shapeO,'.','MarkerSize',24);
    plot(shapeOrotated,'o','MarkerSize',12);
    plot(outO,'o','MarkerSize',12);    
    plot(outOr,'ro','MarkerSize',12);
    
    legend('shape train','shape train rotated','net train output',...
        'shape test','shape test rotated', 'net test output','real net test output');
    grid minor; grid;
    boldify;

    
    %----------------------------------------------------------------------
    % nonlinear volterra series
    % n(t) = a(t) + sum beta * a(t-k) a(t-l)
    at = randn(1,100000) + 0*1i*randn(1,100000);
    L = length(at);
    %beta = [ 0.3 + 0.7*1i, 0.7 - 0.3*1i]; lags = { [1,2] , [1,3] };     
    beta = [ 0.3 + 0*0.7*1i]; lags = { [1,2] };     
    numlags = max(max(lags{:}))+1; att = zeros(numlags,L);
    for ll=0:numlags-1;
        att(ll+1, ll + (1:L-ll) ) = at(1:L-ll);
    end
    
    nt = at; 
    txtvt='Volterra process a~N(0,1): n(0) = a(0)';
    for term = 1:length(beta)
        nt = nt + beta(term) * ...
                att( 1+lags{term}(1),:).*att( 1+lags{term}(2),:);
        txtvt = sprintf('%s + (%0.3f + i%0.3f) a(-%d) a(-%d)',txtvt, ...
                    real(beta(term)),imag(beta(term)),lags{term}(1),lags{term}(2));
    end
    
    L = length(nt);
    numlags = 6+1; ntt = zeros(numlags,L);
    for ll=0:numlags-1;
        ntt(ll+1, ll + (1:L-ll) ) = nt(1:L-ll);
    end    

    traininds = 1:L/2; testinds=(L/2+1):L;    
    out = nt(traininds);
    nttpast = ntt(2:end,traininds);    
    out1 = nt(testinds);
    nttpast1 = ntt(2:end,testinds);

    % linear predictor
    %betahat = conj( (nttpast*nttpast') \ (nttpast * out') );
    %outlinear = betahat'*nttpast;
    params.hiddenSize = [];
    params.layersFcn = 'purelin';params.outputFcn='purelin';    
    params.trainFcn = 'adam';
    params.initFcn = 'nguyen-widrow';
    params.minbatchsize = numel(traininds)/10;
    net = complexnet(params);
    net = net.train(nttpast,out);
    outlinear1 = net.test(nttpast1);
            
    % neural networks parameters
    params=[];
    params.initial_lrate=1e-1;    
    %params.hiddenSize = [64 64 6];
    params.hiddenSize = [16 6]*2;
    params.trainFcn = 'trainlm'; params.minbatchsize = numel(traininds)/10;
    %params.trainFcn = 'sgdmomentum'; params.minbatchsize = numel(traininds)/10;
    %params.trainFcn = 'grad'; params.minbatchsize = numel(traininds)/10;
    params.initFcn = 'nguyen-widrow';
    params.layersFcn = 'mytansig'; 'sigrealimag2';%'sigrealimag2';%'myasinh'
    params.outputFcn = 'purelin';
    params.nbrofEpochs = 300;
    txtml = sprintf(' complex ML activation:%s layers:[%s]',params.layersFcn,num2str(params.hiddenSize));
    cnet = complexnet(params);        
    cnet = cnet.train(nttpast,out);    
    outhat1 = cnet.test(nttpast1);
    
    % re/im MATLAB network
    net = feedforwardnet( params.hiddenSize);
    realifyfn = @(x) [real(x); imag(x)];
    %net = train(net,realifyfn(nttpast),realifyfn(out));
    %outri = net(realifyfn(nttpast1)); outri=outri(1,:)+1i*outri(2,:);        
    net.trainFcn='trainlm';%'traincgf';'traingdx';
    net = train(net,real(nttpast),real(out));
    outri = net(real(nttpast1));    
        
    %{
    % check against matlab weights
    w = net.IW{1,1}; b=net.b{1};
    if size(cnet.Weights{1}) ~= size([w b])
        fprintf('not matching\n');
    else
        cnet.Weights{1} = [w b];
    end
    for ll = 2:cnet.nbrofLayers
        w = net.LW{ll,ll-1} 
        b = net.b{ll}    
        if size(cnet.Weights{ll}) ~= size([w b])    
            fprintf('not matching\n');
        else
            cnet.Weights{ll} = [w b];
        end
    end    
    outhat1 = cnet.test(nttpast1);
    cnet.initFcn = 'previous'; 
    cnet = cnet.train(nttpast,out);    
    outhat1 = cnet.test(nttpast1);
    %}
    
    figure(1234); clf;    
    legtxt={};
    plot(real(out1-at(testinds)),'.-','MarkerSize',24,'LineWidth',2);
    legtxt{end+1}='Volterra process - a(0)';
    hold on;
    plot(real(outhat1),'o-','MarkerSize',12,'LineWidth',2); 
    legtxt{end+1} = 'complex ML output';
    plot(real(outlinear1),'o','MarkerSize',12,'LineWidth',2)
    legtxt{end+1} = 'linear output';
    plot(real(outri),'.-','MarkerSize',12,'LineWidth',2)
    legtxt{end+1} = 're/im ML output';
    
    legend(legtxt,'FontSize',24,'FontWeight','bold');
    xlim([1 200]);
    xlabel('sample number','FontSize',24,'FontWeight','bold');
    ylabel('Real part of samples','FontSize',24,'FontWeight','bold');
    title(sprintf('%s\n %s',txtvt,txtml),'FontSize',24,'FontWeight','bold');
    ax=gca; ax.FontSize=16;
    grid minor; grid;
    boldify;
    
    
    fprintf('mse outhat1 %f outri %f outlinear %f\n',...
        mean(abs(out1-outhat1).^2), ...
        mean(abs(out1-outri).^2), ...
        mean(abs(out1-outlinear1).^2));

    fprintf('mse (less at) outhat1 %f outri %f outlinear %f\n',...
        mean(abs(out1-at(testinds)-outhat1).^2), ...
        mean(abs(out1-at(testinds)-outri).^2), ...
        mean(abs(out1-at(testinds)-outlinear1).^2));
       
    
    
    %----------------------------------------------------------------------
    %}
    
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
        
        Weights      % cell array with the weights in each layer
        SumDeltaWeights   % accumulated gradient
        SumDeltaWeights2  % accumulated |gradient|^2 = gradient .* conj(gradient)

        trainFcn     % 'adam', 'trainlm'=Levenberg-Marquardt, 'gradient'
        initFcn      % 'randn','nguyen-widrow'

        minbatchsize
        batchtype    %'randperm','fixed'
        
        initial_lrate % initial step size for the gradient  
        nbrofEpochs
        
        % mapminmax
        inputSettings
        outputSettings
        
    end
    
    properties (Constant)
        nbrofEpochsdefault = 1e3; % number of iterations picking a batch each time and running gradient
        printmseinEpochs = 1; %50;  % one print per printmeseinEpochs
        beta1=0.9;         % Beta1 is the decay rate for the first moment
        beta2=0.999;       % Beta 2 is the decay rate for the second moment
        
        initial_lratedefault = 1e-2; % initial step size for the gradient        

        initial_mu = 1e-2; % Hessian + mu * eye for Levenberg-Marquardt
        mu_inc = 10;
        mu_dec = 1/10;
        mu_max = 1e10
        
        load = 1e-8;       % loading in denominator of DeltaW for stability
               
        batchsize_per_feature = 50;    % number of samples per feature to use at a time in a epoch
        minbatchsizedefault = 25;
        batchtypedefault = 'randperm';
        
        epochs_drop = 50;   % number of epochs before dropping learning rate
        drop = 0.5;           % drop in learning rate new = old*drop
    end        
        
    methods (Static)        
        function y=realify(x)
            y=[real(x); imag(x)];
        end
        function x=unrealify(y)
            x=y(1:end/2,:) + 1i*y(end/2+1:end,:);
        end
        
        
        % complex randn for intializing weights
        function z=crandn(m,n)
            z=complex(randn(m,n),randn(m,n));
        end        
        function z=crand(m,n)
            z=complex(2*rand(m,n)-1,2*rand(m,n)-1);
        end        
        
        % out = in linear
        function [z,dz]=purelin(x)
            z=x;
            dz=ones(size(x));
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
            cval = 2; kval = 2; lval = -1;
            z =  cval./(1+exp(-kval*x));
            dz = kval * (z.*(1-z/cval));            
            z = z  + lval;
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
            cval = 2; kval = 2;
            zr = ( cval./(1+exp(-kval*real(x))) );
            zi = ( cval./(1+exp(-kval*imag(x))) );            
            z = (zr -1) + 1i*(zi -1);
            dz.real = kval * zr.*(1-zr/cval);
            dz.imag = kval * zi.*(1-zi/cval);
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
                        
            if isfield(params,'initial_lrate')
                obj.initial_lrate= params.initial_lrate; 
            else
                obj.initial_lrate = obj.initial_lratedefault;
            end
            
            if isfield(params,'nbrofEpochs')
                obj.nbrofEpochs= params.nbrofEpochs;
            else
                obj.nbrofEpochs = obj.nbrofEpochsdefault;
            end
            
            % type of training
            if ~isfield(params,'initFcn'), params.initFcn='nguyen-widrow'; end
            obj.initFcn = params.initFcn;
            if ~isfield(params,'trainFcn'), params.trainFcn='adam'; end
            obj.trainFcn = params.trainFcn;
            
            % hidden layers sizes is a vector 
            obj.hiddenSize = params.hiddenSize;
            obj.layers = cell(1,length(obj.hiddenSize)+1);

            outputFcn = params.outputFcn;
            layersFcn = params.layersFcn;
            
            % hidden layers
            for ll=1:length(obj.layers)-1
                obj.layers{ll}.transferFcn = layersFcn;                 
                % each weight matrix is output size x input size since it
                % is applied on the left of the input vector
                % obj.Weights{ll} * x{ll}                
            end
            % output layer
            obj.layers{end}.transferFcn = outputFcn;            
            obj.nbrofLayers = numel(obj.hiddenSize) + 1;            
            [obj.SumDeltaWeights, obj.SumDeltaWeights2] = deal(cell(1,obj.nbrofLayers));            
            switch obj.initFcn
                case 'previous'
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
                                    
            % pick batch size number of sample if possible
            nbrofSamplesinBatch =  max(obj.batchsize_per_feature*nbrofInUnits,obj.minbatchsize);
            nbrofSamplesinBatch =  min(nbrofSamplesinBatch,nbrofSamples);
                        
            w = obj.crandn(nbrofOutUnits,nbrofInUnits);
            b = obj.crandn(1,1);
            mse = -1*ones(1,obj.nbrofEpochs);
            learningRate = obj.initial_lrate;
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
                    case 'grad'
                        % Gradient descent
                        wnew = w - learningRate * dw;
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
                        
            if obj.nbrofLayers>Inf
                % scale the input
                ins = mapminmax.apply(obj.realify(in),obj.inputSettings);
                in = obj.unrealify(ins);
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
                        
            if obj.nbrofLayers>Inf
                % network matches to -1,1 but output is back to original scale
                out = mapminmax.reverse( obj.realify(y{obj.nbrofLayers}),obj.outputSettings);      
                out = obj.unrealify(out);                
            else
                out = y{obj.nbrofLayers};
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
        function obj = train(obj,in,out)
            % in  is features x number of training samples
            % out is desired output x number of training samples            
            
            if obj.nbrofLayers>Inf
                % per feature normalization into -1 1
                [in_normalized,obj.inputSettings] = mapminmax(obj.realify(in));
                [out_normalized,obj.outputSettings] = mapminmax(obj.realify(out));
                in_normalized = obj.unrealify(in_normalized);
                out_normalized = obj.unrealify(out_normalized);                
            else
                in_normalized =in;
                out_normalized = out;
            end
            
            [nbrofInUnits, nbrofSamples] = size(in);
            [nbrofOutUnits, nbr] = size(out);
            if nbr~=nbrofSamples
                error('input and output number of samples must be identical\n');
            end            
            nbrOfNeuronsInEachHiddenLayer = obj.hiddenSize;
             
            % pick batch size number of sample if possible
            nbrofSamplesinBatch =  max(obj.batchsize_per_feature*nbrofInUnits,obj.minbatchsize);
            nbrofSamplesinBatch =  min(nbrofSamplesinBatch,nbrofSamples);            
      
            % allocate space for gradients
            % deltax{n} = dcost/dx{n} vector of dimension x{n}
            % DeltaW{n} = dcost/dWeights{n} matrix of dimension W{n}
            [deltax,DeltaW] = deal(cell(1,obj.nbrofLayers));

            switch obj.trainFcn                        
                case 'trainlm'            
                    [Deltaf] = deal(cell(1,obj.nbrofLayers));
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
                        obj.Weights{nn} = obj.crandn( obj.nbrofUnits(2,nn), obj.nbrofUnits(1,nn) );                        
                    case 'c-nguyen-widrow'
                        %Nguyen-Widrow Algorithm
                        %Initialize all weight of hidden layers with random values
                        %For each hidden layer{
                        %beta = 0.7 * Math.pow(hiddenNeurons, 1.0 / number of inputs);
                        %For each synapse, For each weight, w= w/norm(w) * beta
                        obj.Weights{nn} = obj.crand(  obj.nbrofUnits(2,nn), obj.nbrofUnits(1,nn) );
                        beta = 0.7 * obj.nbrofUnits(2,nn).^(1/obj.nbrofUnits(1,nn));
                        obj.Weights{nn} = beta * obj.Weights{nn}./abs(obj.Weights{nn});    

                    %-----------------------------
                    % non-complex initializations                                        
                    case 'randn'
                        obj.Weights{nn} = randn( obj.nbrofUnits(2,nn), obj.nbrofUnits(1,nn) );                        
                    case 'nguyen-widrow'
                        %Nguyen-Widrow Algorithm
                        %Initialize all weight of hidden layers with random values
                        %For each hidden layer{
                        %beta = 0.7 * Math.pow(hiddenNeurons, 1.0 / number of inputs);
                        %For each synapse, For each weight, w= w/norm(w) * beta
                        obj.Weights{nn} = 2*rand(  obj.nbrofUnits(2,nn), obj.nbrofUnits(1,nn) )-1;
                        beta = 0.7 * obj.nbrofUnits(2,nn).^(1/obj.nbrofUnits(1,nn));
                        obj.Weights{nn} = beta * obj.Weights{nn}./abs(obj.Weights{nn});    
                        
                        
                    % weights already initialized                        
                    case 'previous'
                        if any( size(obj.Weights{nn})~= [obj.nbrofUnits(2,nn), obj.nbrofUnits(1,nn)])
                            error('change obj.initFcn=%s, since previous weights dimensions not matching');
                        end
                end
                
                [DeltaW{nn}, obj.SumDeltaWeights{nn}, obj.SumDeltaWeights2{nn}] = ...
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
            
            switch obj.trainFcn
                case 'trainlm'
                    totalnbrofParameters = sum(obj.nbrofWeights+obj.nbrofBias);                    
                    Hessian = zeros(totalnbrofParameters,totalnbrofParameters);
                    Jac = zeros(nbrofOutUnits,totalnbrofParameters);
                    jace = zeros(totalnbrofParameters,1);
            end
            
            % update the weights over the epochs
            [msetrain,msetest] = deal(-1*ones(1,obj.nbrofEpochs));
            lrate = obj.initial_lrate;
                            
            batchtrain = randperm(nbrofSamples,nbrofSamplesinBatch);
            batchtest = setdiff(1:nbrofSamples,batchtrain);       
            epoch=0;
            while epoch < obj.nbrofEpochs
                epoch=epoch+1;
                % pick a batch for this epoch
                switch obj.batchtype
                    case 'randperm'
                        % pick a batch for this epoch
                        batchtrain = randperm(nbrofSamples,nbrofSamplesinBatch);
                        batchtest = setdiff(1:nbrofSamples,batchtrain);                        
                    case 'fixed'
                        %batch stays the same, so mse can be examined for
                        %changes
                end

                % step in learning rate over epochs
                lrate = obj.initial_lrate * obj.drop^floor((epoch)/obj.epochs_drop);
                
                t = out_normalized(:,batchtrain);           % desired y{nbrofLayers}
                q = out(:,batchtrain);           
                                
                % evaluate network and obtain gradients and intermediate values
                % y{end} is the output
                [curr,x,y0,y,yprime] = obj.test( in(:,batchtrain) );
                
                % mean squared error in the unnormalized
                % not taking 1/2 of the square 
                % normalized would be y{obj.nbrofLayers} - out_normalized
                msecurr = mean( abs(curr(:)-q(:)).^2 );
                                
                %----------------------------------------------------------
                % backpropagate to get all layers                
                splitrealimag = zeros(1,obj.nbrofLayers);
                for nn=obj.nbrofLayers:-1:1                             
                    % deltax is the gradient at the middle of the layer
                    % deltax = d{cost}/d{xn} called "sensitivity"
                    % 
                    deltax{nn} = zeros( obj.nbrofUnits(2,nn), nbrofSamplesinBatch);
                    
                    if isstruct(yprime{nn}), splitrealimag(nn) = 1; else, splitrealimag(nn)=0; end
                    
                    if nn==obj.nbrofLayers
                        % for output layer, using mean-squared error
                        % w = w - mu (y-d) f'(net*) x*
                        % Df is nbrofOutUnits x nbrofOutUnits

                        if splitrealimag(nn)
                            deltax{nn} = real(y{nn} - t) .* yprime{nn}.real + ...
                                1i*imag(y{nn} - t) .* yprime{nn}.imag;                            
                            Deltaf{nn} = yprime{nn}.real +1i*yprime{nn}.imag;                            
                        else
                            deltax{nn} = (y{nn} - t) .* conj( yprime{nn} );
                            Deltaf{nn} = conj( yprime{nn} );
                        end                        
                        
                        Df = zeros(nbrofOutUnits,nbrofOutUnits,nbrofSamplesinBatch);
                        for mm=1:nbrofSamplesinBatch                                                        
                            Df(:,:,mm) = Deltaf{nn}(:,mm);
                        end
                        Deltaf{nn} = Df;                        
                    else                        
                        dx_nnplus1 = deltax{nn+1}; % =0, assigned just for dimensions
                        dx_nn = deltax{nn};                        
                        yp_nn = yprime{nn};
                        W_nnplus1 = obj.Weights{nn+1};
                        
                        Df_nnplus1 = Deltaf{nn+1}; % =0, assigned just for dimensions                        
                        Df_nn = Deltaf{nn};                        
                        for mm=1:nbrofSamplesinBatch                            
                            % last column of weight has no effect of dc/x{nn}
                            % since it is due to bias (hence the 1:end-1)
                            % hidden layers
                            % w_ij = w_ij + mu [sum_k( (d_k-y_k) f'(net_k*) w_ki* )]
                            %                           *  f'(net_i*) x_j*                            
                            if splitrealimag(nn)
                                % for 2-layer derivation, see equation(17)
                                % "Extension of the BackPropagation
                                % algorithm to complex numbers" 
                                % by Tohru Nitta                                
                                ypr = yp_nn.real(:,mm);
                                ypi = yp_nn.imag(:,mm);
                                dxr_nnplus1 = real(dx_nnplus1(:,mm));
                                dxi_nnplus1 = imag(dx_nnplus1(:,mm));                                                                                                  
                                dx = ypr.*(real(W_nnplus1(:,1:end-1)).' * dxr_nnplus1) +...
                                    ypr.*(imag(W_nnplus1(:,1:end-1)).' * dxi_nnplus1) +...
                                    -1i*(...
                                    ypi.*(imag(W_nnplus1(:,1:end-1)).' * dxr_nnplus1) +...
                                    ypi.*(real(W_nnplus1(:,1:end-1)).' * dxi_nnplus1)...
                                    );                                                                             

                                Dfr_nnplus1 = real(Df_nnplus1(:,:,mm));
                                Dfi_nnplus1 = imag(Df_nnplus1(:,:,mm));                                                                   
                                Df = diag(ypr)*(real(W_nnplus1(:,1:end-1)).' * Dfr_nnplus1) +...
                                    diag(ypr)*(imag(W_nnplus1(:,1:end-1)).' * Dfi_nnplus1) +...
                                    -1i*(...
                                    diag(ypi)*(imag(W_nnplus1(:,1:end-1)).' * Dfr_nnplus1) +...
                                    diag(ypi)*(real(W_nnplus1(:,1:end-1)).' * Dfi_nnplus1)...
                                    );                                                                             
                            else
                                % last column of Weights are from the bias
                                % term for nn+1 layer, which does not
                                % contribute to nn layer, hence 1:end-1                                
                                dx = conj(yp_nn(:,mm)).* ( W_nnplus1(:,1:end-1).' * dx_nnplus1(:,mm));                                
                                Df = diag(conj(yp_nn(:,mm))) * ( W_nnplus1(:,1:end-1).' * Df_nnplus1(:,:,mm));
                            end
                            
                            if any(size(dx)~=size(dx_nn(:,mm)))
                                dx
                                dx_nn(:,mm)
                                error('gradient dc/dx dimensions mismatch');                                
                            end
                            dx_nn(:,mm) = dx;
                            Df_nn(:,:,mm) = Df;                            
                        end
                        deltax{nn} = dx_nn;
                        Deltaf{nn} = Df_nn;
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
                end
                
                
                %for linear single neuron 
                % DeltaW = (y{nn} - t) .* conj( yprime{nn} ) * ynnminus1'/nbrofSamplesinBatch;      
                % Jac = conj( yprime{nn} ) * conj(ynnminus1(1:end-1,mm)));
                % jace += transpose(Jac) * (y{obj.nbrofLayers}(:,mm) - t(:,mm));
                
                % update all the weight matrices (including bias weights)
                %
                switch obj.trainFcn
                    case 'trainlm'
                        % cumulative number of parameters to keep track of
                        % where the weights and bias will be kept
                        cP = [0 cumsum(obj.nbrofWeights + obj.nbrofBias)];
                        for mm=1:nbrofSamplesinBatch
                            for nn=obj.nbrofLayers:-1:1
                                if nn==1
                                    ynnminus1 = y0; % no previous layer for first layer
                                else
                                    ynnminus1 = y{nn-1};
                                end
                                layerweightinds = cP(nn) + (1:obj.nbrofWeights(nn));
                                layerbiasinds =  (cP(nn) + obj.nbrofWeights(nn)) + (1:obj.nbrofBias(nn));
                                Df = squeeze( Deltaf{nn}(:,:,mm) );
                                Jac(:,layerweightinds) = kron(Df,conj(ynnminus1(1:end-1,mm)));
                                Jac(:,layerbiasinds) = Df;
                            end
                            if splitrealimag(nn)
                                jace = jace + ...
                                    transpose(real(Jac))*real(y{obj.nbrofLayers}(:,mm) - t(:,mm)) + ...
                                    -1i*transpose(imag(Jac))*imag(y{obj.nbrofLayers}(:,mm) - t(:,mm));                                
                                Hessian = Hessian + transpose(real(Jac))*real(Jac) + ...
                                    -1i*transpose(imag(Jac))*imag(Jac);
                            else
                                jace = jace + ...
                                    transpose(Jac)*(y{obj.nbrofLayers}(:,mm) - t(:,mm));
                                Hessian = Hessian + transpose(Jac)*(Jac);
                            end
                        end
                        jace = jace/nbrofSamplesinBatch;
                        Hessian = Hessian/nbrofSamplesinBatch;
                        
                        q = out(:,batchtrain);
                        msetrn = inf; numstep = 1;
                        while msetrn>msecurr
                            if numstep>1
                                mu=mu*obj.mu_inc; % pad the Hessian more
                                obj=objcurr;      % try again with original weights
                            end
                            ee = eye(size(Hessian));
                            objcurr = obj;
                            
                            % Levenberg Marquardt
                            Hblend = Hessian + mu*ee;%diag(diag(Hessian));
                            
                            if isnan(rcond(Hblend)) || mu>obj.mu_max
                                figure(101); imagesc(abs(Hessian)); colorbar;
                                return; 
                            end
                            
                            %Training stops when any of these conditions occurs:
                            %The maximum number of epochs (repetitions) is reached.
                            %The maximum amount of time is exceeded.
                            %Performance is minimized to the goal.
                            %The performance gradient falls below min_grad.
                            %mu exceeds mu_max.
                            %Validation performance has increased more than max_fail times since the last time it decreased (when using validation).
                            
                            WbWb = Hblend\jace;                            
                            for nn=obj.nbrofLayers:-1:1
                                layerweightinds = cP(nn) + (1:obj.nbrofWeights(nn));
                                layerbiasinds =  (cP(nn) + obj.nbrofWeights(nn)) + (1:obj.nbrofBias(nn));
                                DW = WbWb(layerweightinds,:); 
                                DW = reshape(DW,obj.nbrofUnits(2,nn),obj.nbrofUnits(1,nn)-1);
                                DeltaW{nn} = [DW WbWb(layerbiasinds,:)];
                                obj.Weights{nn} = obj.Weights{nn} -  DeltaW{nn};
                            end
                            curr = obj.test( in(:,batchtrain) );
                            % Check the mse for the update
                            msetrn = mean( abs(curr(:)-q(:)).^2 );
                            fprintf('mu %e msetrn %f\n',mu,msetrn);
                            if isnan(msetrn)
                                any(isnan(objcurr.Weights{1}))
                                any(isnan(obj.Weights{1}))
                            end
                            numstep=numstep+1;
                        end                        
                        % as mu decreases, becomes Newton's method
                        mu=mu*obj.mu_dec;
                end
                
                for nn=1:obj.nbrofLayers                    
                    switch obj.trainFcn
                        case 'grad'
                            % vanilla gradient
                            obj.Weights{nn} = obj.Weights{nn} - lrate * DeltaW{nn};    
                        case 'sgdmomentum'
                            % momentum
                            obj.SumDeltaWeights{nn} = obj.SumDeltaWeights{nn} *obj.beta1 + ...
                                DeltaW{nn} * (1-obj.beta1);
                            % account for the bias due to moving average
                            mhat = obj.SumDeltaWeights{nn} / (1-obj.beta1^epoch);
                            obj.Weights{nn} = obj.Weights{nn} - lrate *  mhat;
                        case 'adam' 
                            % Adam (Adaptive Moment Estimation) takes the best of both
                            % worlds of Momentum and RMSProp
                            %
                            % sum_of_gradient = previous_sum_of_gradient * beta1 + gradient * (1 - beta1) [Momentum]
                            % sum_of_gradient_squared = previous_sum_of_gradient_squared * beta2 + gradient² * (1- beta2) [RMSProp]
                            % delta = -learning_rate * sum_of_gradient / sqrt(sum_of_gradient_squared)
                            % theta += delta
                            % Beta1 is the decay rate for the first moment,
                            % sum of gradient (aka momentum), commonly set at 0.9.
                            % Beta 2 is the decay rate for the second moment,
                            % sum of gradient squared, and it is commonly set at 0.999.
                                                        
                            % momentum
                            obj.SumDeltaWeights{nn} = obj.SumDeltaWeights{nn} *obj.beta1 + ...
                                DeltaW{nn} * (1-obj.beta1);                            
                            % complex Adam modification is that DeltaW.^2
                            % becomes DeltaW.*conj(DeltaW)
                            obj.SumDeltaWeights2{nn} = obj.SumDeltaWeights2{nn}*obj.beta2 + ...
                                DeltaW{nn}.*conj(DeltaW{nn}) * (1-obj.beta2);                            
                            % account for the bias due to moving average
                            %mhat = obj.SumDeltaWeights{nn} / (1-obj.beta1^epoch);
                            %vhat = obj.SumDeltaWeights2{nn} / (1-obj.beta2^epoch);
                            mhat = obj.SumDeltaWeights{nn};
                            vhat = obj.SumDeltaWeights2{nn};
                            obj.Weights{nn} = obj.Weights{nn} - lrate *  mhat./ sqrt(vhat + obj.load);
                    end                    
                end
                
                % Check the mse for the update
                q = out(:,batchtrain);           
                curr = obj.test( in(:,batchtrain) );
                msetrn = mean( abs(curr(:)-q(:)).^2 );            
                
                % Check the mse for the update
                q = out(:,batchtest);           
                curr = obj.test( in(:,batchtest) );
                msetst = mean( abs(curr(:)-q(:)).^2 );                                
                                             
                msetrain(epoch)= msetrn;
                msetest(epoch) = msetst;
                
                if (epoch/obj.printmseinEpochs == floor(epoch/obj.printmseinEpochs))
                    fprintf('mse(%d) train msecurr %0.3f msetrn %0.3f msetest %0.3f lrate %e \n',...
                        epoch,(msecurr),(msetrn),(msetst),lrate);
                end
                
            end % while epoch            
            fprintf('mse(%d) train curr %0.3f new %0.3f test %0.3f lrate %e\n',...
                epoch,(msecurr),(msetrn),(msetst),lrate);
        end
        
        
    end
end

