
rng(1010);

%DEFINE A SIMPLE PROBLEM
in = (randi(2,2,10)-1.5)/2;
%out = 2200*in(1,:).*in(2,:) +10010;
out = in(1,:).*in(2,:)/100;

x = rand(2,1000)*10 - 5 ;
in = x/10;
out = x(1,:).^2 + x(2,:).^2;

L= length(in);
trainInd = 1:floor(0.7*L);
valInd = trainInd(end) + (1:floor(0.15*L));
testInd = (valInd(end) +1):L;

% neural networks parameters
Hidden_Neurons = 20;
params.domap = 1;
params.hiddenSize = [Hidden_Neurons];
params.debugPlots=0;
params.mu = 1e-3;
params.trainFcn = 'trainlm'; params.minbatchsize = numel(trainInd);
params.batchtype='fixed';
params.layersFcn = 'mytansig';
params.outputFcn = 'purelin';
params.initFcn = 'previous';
params.nbrofEpochs = 300;

weightRecord={};
weightRecord2={};
jeRecord={};
jjRecord={};

net = feedforwardnet( params.hiddenSize );

% calls initwb to initialize weights
net = configure(net,in,out);
net = init(net);
IW=net.IW; LW=net.LW; b=net.b; % grab the initial values of the weights

net.trainFcn='trainlm';%'traincgf';'traingdx';
%net.divideFcn='dividerand';
net.divideFcn='divideind';
net.divideParam.trainInd=trainInd;
net.divideParam.valInd=valInd;
net.divideParam.testInd=testInd;

net = train(net,in,out);
length(weightRecord)

cnet = complexnet(params);
cnet = copynet(cnet,net,IW,LW,b,weightRecord,jeRecord,jjRecord,weightRecord2);
cnet.debugCompare = 0
cnet = cnet.train(in,out);


cnet1 = complexnet1(params);
cnet1.initFcn = 'nguyen-widrow';
cnet1=cnet1.train(in,out);


