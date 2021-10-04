% some examples of use of complexcascadenet class

% gradient descent steps work well for many problems, but tend to fail for
% function fitting problems.  the gradient descent approaches here were
% slighly modified for complex operation
addpath ./gradient-descent/


% typical use of machine learning is to realify the input and output
% so that the network operates on real samples.  then, unrealify tho output
% to create complex samples
realifyfn = @(x) [real(x); imag(x)];
unrealifyfun = @(y) y(1:end/2,:) + 1i*y(end/2+1:end,:);

havestatbox = license('test', 'statistics_toolbox');



%% check for single complex layer

params =[];
params.hiddenSize=[]; 
params.outputFcn= 'purelin';
params.trainFcn='trainlm'; %'Adam2','Vanilla';

% map the input values into (-1,1) or smaller so that e.g. tanh and other
% nonlinear activation functions can be used
% can map the magnitude, or the real and imag separately
params.domap = 0;  % 'reim';'gain';

% make a complex cascade net
cnet = complexcascadenet(params);


% random input
x=randn(1,100)+1i*randn(1,100);

% linear output
y = x*(3+5*1i) + (4-2*1i);

% train and print out the weights and bias
cnet = cnet.train(x,y);
cnet.InputWeights
cnet.LayerWeights
cnet.bias


if havestatbox    
    % if you have the NN toolbox, this is matlab's real version of the same
    net = cascadeforwardnet(params.hiddenSize);
    net = train(net,realifyfn(x),realifyfn(y));
    net.IW
    net.LW
    net.b
end



%% rotation example 
% see Nitta's "An Extension of the Back-Propagation Algorithm to Complex Numbers"

% ellipse
% I
%rot = pi/1000000; 
rot = pi/4;
R = [cos(rot) -sin(rot); sin(rot) cos(rot)];
R = diag([0.5 0.3])*R;

% make an I in the imaginary plane
% NOTE: making num large increases the number of training samples and
% allows the real network to learn the re/im rotation whereas the complex
% network learns it with much less training
num = 100;
ini1 = linspace(-0.945,0.945,round(2*num)); inr1 = zeros(size(ini1));
inr2 = linspace(-0.2,0.2,round(2*num)); ini2 = 0.95*ones(size(inr2));
inr3 = inr2; ini3=-ini2;
shapeI=[inr1 inr2 inr3] + 1i*[ini1 ini2 ini3];

% make an O
A = 0.6; B = 0.3; th = linspace(0,2*pi,220);
shapeO = A*cos(th) + 1i*B*sin(th);

% rotate the I and O
shape = shapeI;
y= R * [real(shape); imag(shape)];
shaperotated = y(1,:) + 1i*y(2,:);
y= R * [real(shapeO); imag(shapeO)];
shapeOrotated = y(1,:) + 1i*y(2,:);

% define some training, validation and testing data
L= length(shape);
trainInd = 1:floor(0.9*L);
valInd = trainInd(end) + (1:round(0.1*L));
testInd = (valInd(end) +1):L;


% define the parameters of the network that is going to learn
% rotation+scale
params = [];

% try different sizes for the network
%hiddenSize_reim = [2 4 2];
hiddenSize_reim = [1 16 1];
hiddenSize = [1 32 1]; % [1 2 2];


nbrofEpochs = 500;
max_fail = nbrofEpochs;

params.hiddenSize=hiddenSize;
params.max_fail = max_fail;
params.nbrofEpochs = nbrofEpochs;


% best if no mapminmax since I is not symmetric
% but complex also has decent performance
%params.domap='complex';
%params.domap='reim';
params.domap='gain';

params.layersFcn='sigrealimag2'; 
params.outputFcn='purelin';

params.minbatchsize='split90_10'; 
params.batchtype = 'fixed';

params.debugPlots = 0; params.performancePlots = 1;


% can try various approaches
if 1
    params.initFcn = 'crandn';
    params.trainFcn = 'trainlm'; 
elseif 1
    params.initFcn = 'crandn';
    params.trainFcn = 'Adam2'; params.nbrofEpochs=5000;
else
    params.initFcn='crandn';
    params.trainFcn = 'trainbr'; params.nbrofEpochs=2000;
end


%--------------------------------------------------------------------------
% REAL NETWORK
if havestatbox    
    % could try strictly feed forward, or one with skip connects (called
    % cascade)
    %net = feedforwardnet( hiddenSize_reim );
    net = cascadeforwardnet( params.hiddenSize);
    
    % could force a particular architecture to make real nets solve the
    % problem
    DOPARTICULAR = 1;
    if DOPARTICULAR
        trainFcn = 'trainlm';%'traincgf';'traingdx';
        net.trainFcn=trainFcn;
        net.trainParam.max_fail = max_fail;
        net.trainParam.epochs = nbrofEpochs;
        net.trainParam.showCommandLine=true;
        
        % 90/10 doesn't work well for real since it is poorly matched to
        % problem
        %net.divideFcn='divideind';
        %net.divideParam.trainInd=trainInd;
        %net.divideParam.valInd=valInd;
        %net.divideParam.testInd=testInd;
        
        net = configure(net,realifyfn(shape),realifyfn(shaperotated));
        net.initFcn = 'initlay';
        [dim1,dim2] = size(net.inputWeights);
        for ii= 1:dim1
            for jj=1:dim2
                net.inputWeights{ii,jj}.initFcn = 'rands';
            end
        end
        [dim1,dim2] = size(net.inputWeights);
        for ii= 1:dim1
            for jj=1:dim2
                net.layerWeights{ii,jj}.initFcn = 'rands';
            end
        end
        [dim1] = length(net.biases);
        for ii=1:dim1, net.biases{ii}.initFcn = 'rands'; end
        net = init(net);
        
        for ii=1:length(net.layers)
            net.layers{ii}.transferFcn='tansig';
        end
    end
    
    % train the network
    net = train(net,realifyfn(shape),realifyfn(shaperotated));
    outhatr = net(realifyfn(shape)); outhatr=outhatr(1,:)+1i*outhatr(2,:);
    outOr =net(realifyfn(shapeO)); outOr=outOr(1,:)+1i*outOr(2,:);

end
%--------------------------------------------------------------------------




%--------------------------------------------------------------------------
% COMPLEX NETWORK
if 0
    cnet = complexnet(params);
elseif 1
    if 0
        params.inputConnect = [1;0;0];
        params.layerConnect = zeros(3,3);
        params.layerConnect(2,1) = 1;
        params.layerConnect(3,2) = 1;
        params.biasConnect = [1;1;1;];
    end
    cnet = complexcascadenet(params);
end
cnet = cnet.train(shape,shaperotated);
outhat = cnet.test(shape);
outO = cnet.test(shapeO);
%print(cnet)
%--------------------------------------------------------------------------


% for debugging, print out the gain settings for each network
disp('complex net input gain');
cnet.inputSettings.gain
disp('complex net output gain');
cnet.outputSettings.gain
if havestatbox    
    disp('real net input gain');    
    net.inputs{1}.processSettings{1}.gain
    disp('real net output gain');        
    net.outputs{end}.processSettings{1}.gain
end

% print out the mse for the outputs
disp('mse for complex');
norm( shapeOrotated - outO).^2
if havestatbox    
    disp('mse for real');
    norm( shapeOrotated - outOr).^2
end

% make plots
legtxt = {};
figure(123);clf;
plot(shape,'.','MarkerSize',24); legtxt{end+1} = 'shape train';
hold on;
plot(shaperotated,'v','MarkerSize',12); legtxt{end+1} = 'shape train rotated';
plot(outhat,'s','MarkerSize',12); legtxt{end+1} = 'complex net train output';
if havestatbox    
    plot(outhatr,'s','MarkerSize',12);legtxt{end+1} = 'real net train output';
end

plot(shapeO,'.','MarkerSize',24);legtxt{end+1} ='shape test';
plot(shapeOrotated,'h','MarkerSize',12);legtxt{end+1} ='shape test rotated';
plot(outO,'o','MarkerSize',12); legtxt{end+1} =sprintf('%s complex net test output',num2str(hiddenSize));
if havestatbox    
    plot(outOr,'ro','MarkerSize',12); legtxt{end+1} = sprintf('%s real net test output',num2str(hiddenSize_reim));
end

legend(legtxt,'Location','best','FontSize',18);
axis tight;
xlim([-0.6 0.6]);
ylim([-1 1]);
grid minor; grid;
xlabel('real','FontSize',24);
ylabel('imaginary','FontSize',24);
