% env = rlPredefinedEnv("CartPole-Discrete");
env = cMyRLEnv2();
obsInfo = getObservationInfo(env);
actInfo = getActionInfo(env);
net = [
    featureInputLayer(obsInfo.Dimension(1))
    fullyConnectedLayer(20)
    reluLayer
    % fullyConnectedLayer(length(actInfo.Elements))
    fullyConnectedLayer(actInfo.Dimension(1))
    ];
net = dlnetwork(net);
summary(net)
critic = rlVectorQValueFunction(net,obsInfo,actInfo);
agent = rlDQNAgent(critic);
agent.AgentOptions.UseDoubleDQN = false;
agent.AgentOptions.TargetSmoothFactor = 1;
agent.AgentOptions.TargetUpdateFrequency = 4;
agent.AgentOptions.ExperienceBufferLength = 1e5;
agent.AgentOptions.MiniBatchSize = 256;
agent.AgentOptions.CriticOptimizerOptions.LearnRate = 1e-3;
agent.AgentOptions.CriticOptimizerOptions.GradientThreshold = 1;
trainOpts = rlTrainingOptions(...
    MaxEpisodes=1000, ...
    MaxStepsPerEpisode=500, ...
    Verbose=false, ...
    Plots="training-progress",...
    StopTrainingCriteria="AverageReward",...
    StopTrainingValue=480);
trainingStats = train(agent,env,trainOpts);
simOptions = rlSimulationOptions(MaxSteps=500);
experience = sim(env,agent,simOptions);
