clc;clear;close all;
% 환경 만들기
previousRngState = rng(0,"twister");
env = rlPredefinedEnv("CartPole-Discrete");
obsInfo = getObservationInfo(env);
actInfo = getActionInfo(env);

% Agent만들기
rng(0,"twister");
initOpts = rlAgentInitializationOptions(NumHiddenUnit=20);
agentOpts = rlDQNAgentOptions( ...
    MiniBatchSize = 256,...
    TargetSmoothFactor = 1, ...
    TargetUpdateFrequency = 4,...
    UseDoubleDQN = false);
agentOpts.EpsilonGreedyExploration.EpsilonDecay = 1e-3;
agent = rlDQNAgent(obsInfo,actInfo,initOpts,agentOpts);

% Train
rng(0,"twister");
trainOpts = rlTrainingOptions(...
    MaxEpisodes=1000, ...
    MaxStepsPerEpisode=500, ...
    Verbose=false, ...
    Plots="training-progress",...
    StopTrainingCriteria="EvaluationStatistic",...
    StopTrainingValue=500);
evl = rlEvaluator(EvaluationFrequency=20, NumEpisodes=5);
trainingStats = train(agent,env,trainOpts,Evaluator=evl);

% Simulation
rng(0,"twister");
plot(env)
simOptions = rlSimulationOptions(MaxSteps=500);
experience = sim(env,agent,simOptions);
totalReward = sum(experience.Reward)
rng(previousRngState);
