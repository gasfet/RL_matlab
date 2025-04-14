classdef cMyRLEnv2 < rl.env.MATLABEnvironment
    %CMYRLENV Summary of this class goes here
    %   Detailed explanation goes here

    properties
        Target (1,2) double = [3,3]
        State = zeros(2,1)
    end

    properties(Access = protected)
        % Initialize internal flag to indicate episode termination
        IsDone = false
    end

    methods

        function obj = cMyRLEnv2()
            % Initialize observation settings
            ObservationInfo = rlNumericSpec([2 1]);
            ObservationInfo.Name = 'Position';
            ObservationInfo.Description = 'xy 좌표';

            % Initialize action settings
            % ActionInfo = rlFiniteSetSpec([-1 1]);
            ActionInfo = rlNumericSpec([2 1]);
            ActionInfo.Name = 'CartPole Action';

            % The following line implements built-in functions of the RL environment
            obj = obj@rl.env.MATLABEnvironment(ObservationInfo,ActionInfo);

            % Initialize property values and precompute necessary values
            updateActionInfo(obj);
        end

        % Reset environment to initial state and return initial observation
        function InitialObservation = reset(obj)
            InitialObservation = [0;0];
            obj.State = InitialObservation;
            notifyEnvUpdated(obj);
        end

        function [Observation,Reward,IsDone,Info] = step(obj,Action)
            Info = [];

            Action = clip(Action, -100, 100);

            % Unpack state vector
            x = obj.State(1);
            y = obj.State(2);
            Observation = [x+Action(1);y+Action(2)];
            obj.State = Observation;

            % disp(sprintf("x= %2.f, y = %.2f",x,y));
            IsDone = all([abs(x-obj.Target(1)) < 0.01 , abs(y-obj.Target(2)) < 0.01]);
            obj.IsDone = IsDone;

            % Get reward
            Reward = 1/sqrt((x-obj.Target(1))^2 + (y-obj.Target(2))^2);
            % disp(Reward);
            % (Optional) Use notifyEnvUpdated to signal that the
            % environment has been updated (for example, to update the visualization)
            notifyEnvUpdated(obj);
        end

    end


    methods(Access = protected)

        function updateActionInfo(obj)
            %this.ActionInfo.Elements = this.MaxForce*[-1 1];
        end

    end

end
