classdef complexnet < handle
    % complexnet Feed-forward neural network with complex Adam 
    % backpropagation
    %
    % Swaroop Appadwedula Gr.64 
    % August 13,2020 in the time of COVID
    
    %{
    %----------------------------------------------------------------------
    % check for single complex layer    
    params.hiddenSize=[]; params.outputFcn='purelin'; net = complexnet(params)
    x=randn(1,100)+1i*randn(1,100); 
    net.trainsingle(x,x*(3+5*1i) + (4-2*1i))
    net = net.train(x,x*(3+5*1i) + (4-2*1i))    
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
    rot = pi/2; R = [cos(rot) -sin(rot); sin(rot) cos(rot)];  
    R = diag([0.5 0.3])*R;
    
    ini1 = linspace(-0.5,0.5,15); inr1 = zeros(size(ini1));
    inr2 = linspace(-0.1,0.1,5); ini2 = 0.55*ones(size(inr2));
    inr3 = inr2; ini3=-ini2;    
    shapeI=[inr1 inr2 inr3] + 1i*[ini1 ini2 ini3];
        
    A = 0.6; B = 0.3; th = linspace(0,2*pi,220);
    shapeO = A*cos(th) + 1i*B*sin(th);

    shape = shapeI;    
    y= R * [real(shape); imag(shape)];
    shaperotated = y(1,:) + 1i*y(2,:);
    
    y= R * [real(shapeO); imag(shapeO)];
    shapeOrotated = y(1,:) + 1i*y(2,:);
    
    params.hiddenSize=[1 6 1]; 
    params.layersFcn='sigrealimag2'; params.outputFcn='sigrealimag2'; 
    net = complexnet(params)    
    net = net.train(shape,shaperotated);
    [~,~,y,~] = net.test(shape); outhat = y{end};
    [~,~,y,~] = net.test(shapeO); outO = y{end};    
    print(net)        


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
    %}
    
    properties
        hiddenSize   % vector of hidden layer sizes, not including output
                     % empty when single layer                   
        layers       % cell array containing transferFcn spec, etc
                     % layers{ll}.transferFcn
        nbrofLayers  % number of layers hidden + 1 
        nbrofUnits   % 2xnbrofLayers matrix 
                     % with row1=input dim, row2=output dim
        Weights      % cell array with the weights in each layer
        SumDeltaWeights   % accumulated gradient
        SumDeltaWeights2  % accumulated |gradient|^2 = gradient .* conj(gradient)
    end
    
    properties (Constant)
        nbrOfEpochs = 1e4; % number of iterations picking a batch each time and running gradient
        printmseinEpochs = 5; %50;  % one print per printmeseinEpochs
        beta1=0.9;         % Beta1 is the decay rate for the first moment
        beta2=0.999;       % Beta 2 is the decay rate for the second moment
        learningRate = 1e-3; % step size for the gradient        
        load = 1e-8;       % loading in denominator of DeltaW for stability
        batchsize_per_feature = 8;    % number of samples per feature to use at a time in a epoch
    end
        
    methods (Static)        
        % complex rand for intializing weights
        function z=crandn(m,n)
            z=complex(randn(m,n),randn(m,n))/sqrt(2);
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
        function [z,dz]=crelu(x)            
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
        function [z,dz]=sigrealimag2(x)
            zr = ( 1./(1+exp(-real(x))) );
            zi = ( 1./(1+exp(-imag(x))) );
            z = (2*zr -1) + 1i*(2*zi -1);
            dz.real = 2*zr.*(1-zr);
            dz.imag = 2*zi.*(1-zi);
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
            if ~isfield(params,'outputFcn'), params.outputFcn = 'purephase'; end
            if ~isfield(params,'layersFcn'), params.layersFcn = 'purelin'; end

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
            [obj.Weights, obj.SumDeltaWeights, obj.SumDeltaWeights2] = deal(cell(1,obj.nbrofLayers));            
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
                                    
            % pick batch size number of sample if possible
            nbrofSamplesinBatch =  min(obj.batchsize_per_feature*nbrofInUnits,nbrofSamples);
                        
            w = obj.crandn(nbrofInUnits,1);
            b = obj.crandn(1,1);
            mse = -1*ones(1,obj.nbrOfEpochs);
            for epoch=1:obj.nbrOfEpochs                                
                % pick a batch for this epoch
                batch = randperm(nbrofSamples,nbrofSamplesinBatch);                
                y = in(:,batch);
                t = out(:,batch);           % desired y{nbrofLayers}                
                netval = transpose(w)*y + b;
                [curr,termi] = fn( netval );
                
                if splitrealimag
                    err = real(t - curr).*termi.real + 1i* imag(t - curr).*termi.imag;
                else
                    err = (t - curr).*conj(termi);
                end
                dw =  transpose( err*y' )/nbrofSamplesinBatch;
                w = w + obj.learningRate * dw;
                d =  mean(  err );
                b = b + d;                                   
                
                mse(epoch) = (t-curr)*(t-curr)'/nbrofSamplesinBatch;
                
                if (epoch/obj.printmseinEpochs == floor(epoch/obj.printmseinEpochs))
                    fprintf('%s mse(%d) %0.3f \n',transferFcn,epoch,mse(epoch));
                end
            end
            Weights = [w; b];
            outhat = fn( transpose(w)*in + b);
        end
        
        %----------------------------------------------------------
        % given an input, determine the network output and intermediate
        % terms
        function [x,y0,y,yprime] = test(obj,in)
            
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
                    
                    % do include the biasvec as part of y
                    y{nn} = [y{nn}; biasvec];
                end
                ynnminus1 = y{nn};
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

            %if any(~isreal(in)) || any(~isreal(out))
            %    error('this is the real version made for testing');
            %end
            
            [nbrofInUnits, nbrofSamples] = size(in);
            [nbrofOutUnits, nbr] = size(out);
            if nbr~=nbrofSamples
                error('input and output number of samples must be identical\n');
            end            
            nbrOfNeuronsInEachHiddenLayer = obj.hiddenSize;
            
            % pick batch size number of sample if possible
            nbrofSamplesinBatch =  min(obj.batchsize_per_feature*nbrofInUnits,nbrofSamples);
            
            % allocate space for gradients
            % deltax{n} = dcost/dx{n} vector of dimension x{n}
            % DeltaW{n} = dcost/dWeights{n} matrix of dimension W{n}
            [deltax,DeltaW] = deal(cell(1,obj.nbrofLayers));

            % once a input is provided, dimensions can be determined
            % initialize the weights
            % this matrix indicates the input (row1) and output (row2)
            % dimensions            
            obj.nbrofUnits(1,:) = [nbrofInUnits  nbrofInUnits*nbrOfNeuronsInEachHiddenLayer];  % input size
            obj.nbrofUnits(2,:) = [nbrofInUnits*nbrOfNeuronsInEachHiddenLayer nbrofOutUnits];  % output size             
                        
            % include a bias to allow for functions away from zero
            % bias changes the input dimension since weigths are applied to
            % bias as well, but it does not change output dimension
            obj.nbrofUnits(1,1:end) =  obj.nbrofUnits(1,1:end) + 1;    % all input sizes of layers incremented                   
            
            for nn=1:obj.nbrofLayers                    
                % since y = W*x, the dimensions of W are out x in 
                % NOT in x out                                
                obj.Weights{nn} = obj.crandn( obj.nbrofUnits(2,nn), obj.nbrofUnits(1,nn) );                   
                [DeltaW{nn}, obj.SumDeltaWeights{nn}, obj.SumDeltaWeights2{nn}] = ...
                    deal(zeros( obj.nbrofUnits(2,nn), obj.nbrofUnits(1,nn) ));   
            end            
           
            % update the weights over the epochs
            mse = -1*ones(1,obj.nbrOfEpochs);
            for epoch = 1:obj.nbrOfEpochs
                
                % pick a batch for this epoch
                batch = randperm(nbrofSamples,nbrofSamplesinBatch);
                
                t = out(:,batch);           % desired y{nbrofLayers}
                
                % evaluate network and obtain gradients and intermediate values
                [x,y0,y,yprime] = obj.test( in(:,batch) );
                
                % mean squared error
                mse(epoch)=0.5*mean( abs(y{obj.nbrofLayers}(:)-t(:)).^2 );
                
                if (epoch/obj.printmseinEpochs == floor(epoch/obj.printmseinEpochs))
                    fprintf('mse(%d) %0.3f\n',epoch,mse(epoch));
                end
                
                %----------------------------------------------------------
                % backpropagate to get all layers                
                for nn=obj.nbrofLayers:-1:1                             
                    % deltax is the gradient at the middle of the layer
                    % deltax = d{cost}/d{xn} called "sensitivity"                                
                    deltax{nn} = zeros( obj.nbrofUnits(2,nn), nbrofSamplesinBatch);
                    
                    if isstruct(yprime{nn}), splitrealimag = 1; else, splitrealimag=0; end
                    
                    if nn==obj.nbrofLayers
                        % for output layer, using mean-squared error
                        % w = w - mu (y-d) f'(net*) x*                        
                        if splitrealimag
                            deltax{nn} = real(y{nn} - t) .* yprime{nn}.real + ...
                                1i*imag(y{nn} - t) .* yprime{nn}.imag;
                        else
                            deltax{nn} = (y{nn} - t) .* conj( yprime{nn} );
                        end                        
                    else
                        for mm=1:nbrofSamplesinBatch 
                            % last column of weight has no effect of dc/x{nn}
                            % since it is due to bias
                            % hidden layers
                            % w_ij = w_ij + mu [sum_k( (d_k-y_k) f'(net_k*) w_ki* )]
                            %                           *  f'(net_i*) x_j*
                            
                            if splitrealimag
                                ypr = yprime{nn}.real(:,mm);
                                ypi = yprime{nn}.imag(:,mm);
                                dxr_nnplus1 = real(deltax{nn+1}(:,mm));
                                dxi_nnplus1 = imag(deltax{nn+1}(:,mm));
                                dx = transpose(  (obj.Weights{nn+1}(:,1:end-1)) *diag(ypr) ) * dxr_nnplus1 + ...
                                    1i* transpose(  (obj.Weights{nn+1}(:,1:end-1)) *diag(ypi) ) * dxi_nnplus1;                                
                            else                                
                                dx = transpose(  (obj.Weights{nn+1}(:,1:end-1)) *diag(conj(yprime{nn}(:,mm)))) * deltax{nn+1}(:,mm);
                            end
                            
                            if any(size(dx)~=size(deltax{nn}(:,mm)))
                                dx
                                deltax{nn}(:,mm)
                                error('gradient dc/dx dimensions mismatch');                                
                            end
                            deltax{nn}(:,mm) = dx;
                        end
                    end
                    

                    if nn==1 
                        ynnminus1 = y0; % no previous layer for first layer
                    else
                        ynnminus1 = y{nn-1}; 
                    end
                    
                    DeltaW{nn} = zeros( obj.nbrofUnits(2,nn), obj.nbrofUnits(1,nn) );
                    for mm=1:nbrofSamplesinBatch
                        onesvec = ones(obj.nbrofUnits(2,nn),1);                                            
                        dW = diag(deltax{nn}(:,mm)) * ( onesvec* ynnminus1(:,mm)' );
                        
                        if any(size(dW)~=size(DeltaW{nn}))
                            dW
                            DeltaW{nn}
                            error('gradient dc/dW dimensions mismatch');                                
                        end
                        DeltaW{nn} = DeltaW{nn} + dW;
                    end                                        
                    DeltaW{nn} = DeltaW{nn}/nbrofSamplesinBatch;
                end               
                
                % update all the weight matrices
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
                
                for nn=1:obj.nbrofLayers
                    obj.SumDeltaWeights{nn} = obj.SumDeltaWeights{nn} *obj.beta1 + DeltaW{nn} * (1-obj.beta1);
                    
                    % complex Adam modification is that DeltaW.^2
                    % becomes DeltaW.*conj(DeltaW)
                    obj.SumDeltaWeights2{nn} = obj.SumDeltaWeights2{nn} *obj.beta2 + DeltaW{nn}.*conj(DeltaW{nn}) * (1-obj.beta2);                    
                    
                    % vanilla gradient
                    %obj.Weights{nn} = obj.Weights{nn} - obj.learningRate * DeltaW{nn};
                    
                    % Adam
                    obj.Weights{nn} = obj.Weights{nn} - ...
                        obj.learningRate * obj.SumDeltaWeights{nn} ./ sqrt(obj.SumDeltaWeights2{nn} + obj.load);                    
                end                                   
            end
            outhat = y{obj.nbrofLayers};
        end
        
        
    end
end

