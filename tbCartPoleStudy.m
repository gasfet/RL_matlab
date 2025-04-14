clc;clear;close all;
% 환경 만들기
previousRngState = rng(0,"twister");
env = rlPredefinedEnv("CartPole-Discrete");
obsInfo = getObservationInfo(env);
actInfo = getActionInfo(env);

% Agent만들기

TypeList = ["DQN", "PG", "AC"];
agType =  TypeList(3);
switch agType
    case "DQN"
        agent = dqnCase(obsInfo, actInfo);
        evl = rlEvaluator(EvaluationFrequency=20, NumEpisodes=5);
        trainOpts = rlTrainingOptions(...
            MaxEpisodes=1000, ...
            MaxStepsPerEpisode=500, ...
            Verbose=false, ...
            Plots="training-progress",...
            StopTrainingCriteria="EvaluationStatistic",...
            StopTrainingValue=500);
    case "PG"
        agent = pgCase(obsInfo, actInfo);
        evl = rlEvaluator(EvaluationFrequency=20, NumEpisodes=5);
        trainOpts = rlTrainingOptions(...
            MaxEpisodes=1000, ...
            MaxStepsPerEpisode=500, ...
            Verbose=false, ...
            Plots="training-progress",...
            StopTrainingCriteria="EvaluationStatistic",...
            StopTrainingValue=500);
    case "AC"
        agent = acCase(obsInfo, actInfo);
        evl = [];
        trainOpts = rlTrainingOptions(...
            MaxEpisodes=1000,...
            MaxStepsPerEpisode=500,...
            Verbose=false,...
            Plots="training-progress",...
            StopTrainingCriteria="AverageReward",...
            StopTrainingValue=480,...
            ScoreAveragingWindowLength=10);
end

% train
if ~exist('agent', 'var'), return; end
% if ~exist('evl', 'var'), evl=[]; end
% trainOpts = rlTrainingOptions(...
%     MaxEpisodes=1000, ...
%     MaxStepsPerEpisode=500, ...
%     Verbose=false, ...
%     Plots="training-progress",...
%     StopTrainingCriteria="EvaluationStatistic",...
%     StopTrainingValue=500);

rng(0,"twister");
% plot(env);
% if isempty(evl)
%     trainingStats = train(agent,env,trainOpts);
% else
    trainingStats = train(agent,env,trainOpts,Evaluator=evl);
% end

% Simulation
rng(0,"twister");
plot(env);
switch agType
    case "PG"
        agent.UseExplorationPolicy = false;
end

simOptions = rlSimulationOptions(MaxSteps=500);
experience = sim(env,agent,simOptions);
totalReward = sum(experience.Reward);
rng(previousRngState);


function agent = dqnCase(obsInfo, actInfo)
    initOpts = rlAgentInitializationOptions(NumHiddenUnit=20);
    agentOpts = rlDQNAgentOptions( ...
        MiniBatchSize = 256,...
        TargetSmoothFactor = 1, ...
        TargetUpdateFrequency = 4,...
        UseDoubleDQN = false);
    agentOpts.EpsilonGreedyExploration.EpsilonDecay = 1e-3;

    agent = rlDQNAgent(obsInfo,actInfo,initOpts,agentOpts);


    % trainOpts = rlTrainingOptions(...
    %     MaxEpisodes=1000, ...
    %     MaxStepsPerEpisode=500, ...
    %     Verbose=false, ...
    %     Plots="training-progress",...
    %     StopTrainingCriteria="EvaluationStatistic",...
    %     StopTrainingValue=500);
    % evl = rlEvaluator(EvaluationFrequency=20, NumEpisodes=5);
    % trainingStats = train(agent,env,trainOpts,Evaluator=evl);

end

function agent = pgCase(obsInfo, actInfo)

    actorNet = [
        featureInputLayer(prod(obsInfo.Dimension))
        fullyConnectedLayer(10)
        reluLayer
        fullyConnectedLayer(numel(actInfo.Elements))
        softmaxLayer
        ];
    actorNet = dlnetwork(actorNet);
    actor = rlDiscreteCategoricalActor(actorNet,obsInfo,actInfo);

    % prb = evaluate(actor,{rand(obsInfo.Dimension)});
    % prb{1}

    agent = rlPGAgent(actor);
    % getAction(agent,{rand(obsInfo.Dimension)})
    agent.AgentOptions.ActorOptimizerOptions = ...
        rlOptimizerOptions(LearnRate=5e-3, ...
            GradientThreshold=1);

    % trainOpts = rlTrainingOptions(...
    %     MaxEpisodes=1000, ...
    %     MaxStepsPerEpisode=500, ...
    %     Verbose=false, ...
    %     Plots="training-progress",...
    %     StopTrainingCriteria="EvaluationStatistic",...
    %     StopTrainingValue=500);
    %
    % evl = rlEvaluator(EvaluationFrequency=20, NumEpisodes=10);
    %
    % rng(0,"twister");
    % trainingStats = train(agent,env,trainOpts,Evaluator=evl);
end

function agent = acCase(obsInfo, actInfo)

    criticNet = [
        featureInputLayer(obsInfo.Dimension(1))
        fullyConnectedLayer(32)
        reluLayer
        fullyConnectedLayer(1)
        ];
    criticNet = dlnetwork(criticNet);
    critic = rlValueFunction(criticNet,obsInfo);
    % getValue(critic,{rand(obsInfo.Dimension)})



    actorNet = [
        featureInputLayer(obsInfo.Dimension(1))
        fullyConnectedLayer(32)
        reluLayer
        fullyConnectedLayer(numel(actInfo.Elements))
        softmaxLayer
        ];
    actorNet = dlnetwork(actorNet);
    actor = rlDiscreteCategoricalActor(actorNet,obsInfo,actInfo);

    % prb = evaluate(actor,{rand(obsInfo.Dimension)})

    agent = rlACAgent(actor,critic);
    agent.AgentOptions.EntropyLossWeight = 0.01;
    agent.AgentOptions.ActorOptimizerOptions.LearnRate = 1e-2;
    agent.AgentOptions.ActorOptimizerOptions.GradientThreshold = 1;
    agent.AgentOptions.CriticOptimizerOptions.LearnRate = 1e-2;
    agent.AgentOptions.CriticOptimizerOptions.GradientThreshold = 1;

    % trainOpts = rlTrainingOptions(...
    %     MaxEpisodes=1000,...
    %     MaxStepsPerEpisode=500,...
    %     Verbose=false,...
    %     Plots="training-progress",...
    %     StopTrainingCriteria="AverageReward",...
    %     StopTrainingValue=480,...
    %     ScoreAveragingWindowLength=10);
    %
    % trainingStats = train(agent,env,trainOpts);

end
