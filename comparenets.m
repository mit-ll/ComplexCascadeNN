
% need weight record check how weights are updated in matlab
warning('Move trainlm_editedforweightRecord.m to trainlm.m first');
pause;

rng(101);

compareChoice = 'd';
switch compareChoice
    case 'a'
        in = (randi(2,2,10)-1.5)/2;
        %out = 2200*in(1,:).*in(2,:) +10010;
        out = in(1,:).*in(2,:)/100;
        Hidden_Neurons = [20 5 1];
    case 'b'
        x = rand(2,1000)*371 - 5 ;
        in = x/10;
        out = 13* ( x(1,:).^2 + x(2,:).^2 );

        % two outputs
        out = 13* [ x(1,:).^2; x(2,:).^2 ];

        Hidden_Neurons = [20 5 1];
        %Hidden_Neurons = [1];
        %Hidden_Neurons = [];
    case 'bb'
        x = rand(1,1000)*371 - 5 ;
        in = x/10;
        out = 13* ( x(1,:).^2 );
        Hidden_Neurons = [20 5 1];
        Hidden_Neurons = [1];
    case 'c'
        %4e-8 error
        in = 2*rand(4,100000)-1;
        out = sin(2*pi*in(1,:)) .* in(2,:).^2 .* in(3,:).^3 .* in(4,:).^4 .* ...
            exp(-1*(in(1,:)+in(2,:)+in(3,:)+in(4,:)));
        Hidden_Neurons = [4 50];
    case 'd'
        %load run4;
        in = realifyfn(intrain);
        out = realifyfn(outtrain);
        Hidden_Neurons = [8 4 1];
end


rng(randi(100,1,1));

L= length(in);
trainInd = 1:floor(0.7*L);
valInd = trainInd(end) + (1:floor(0.15*L));
testInd = (valInd(end) +1):L;

% neural networks parameters

params.domap = 'reim';
params.hiddenSize = [Hidden_Neurons];
params.debugPlots=0;
params.mu = 1e-3;
params.initFcn = 'nguyen-widrow';
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
net.trainParam.epochs=5

net = train(net,in,out);
length(weightRecord)

cnet = complexnet(params);
cnet = copynet(cnet,net,IW,LW,b,weightRecord,jeRecord,jjRecord,weightRecord2);
cnet.debugCompare = 1
cnet = cnet.train(in,out);
