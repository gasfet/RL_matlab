classdef cMyRLEnv < rl.env.MATLABEnvironment
    %CMYRLENV Summary of this class goes here
    %   Detailed explanation goes here

    properties
        % Specify and initialize the necessary properties of the environment
        % Acceleration due to gravity in m/s^2
        Gravity = 9.8

        % Mass of the cart
        CartMass = 1.0

        % Mass of the pole
        PoleMass = 0.1

        % Half the length of the pole
        HalfPoleLength = 0.5

        % Max force the input can apply
        MaxForce = 10

        % Sample time
        Ts = 0.02

        % Angle at which to fail the episode (radians)
        AngleThreshold = 12 * pi/180

        % Distance at which to fail the episode
        DisplacementThreshold = 2.4

        % Reward each time step the cart-pole is balanced
        RewardForNotFalling = 1

        % Penalty when the cart-pole fails to balance
        PenaltyForFalling = -10
    end

    properties
        % Initialize system state [x,dx,theta,dtheta]'
        State = zeros(4,1)
    end

    properties(Access = protected)
        % Initialize internal flag to indicate episode termination
        IsDone = false
    end

    methods

        function obj = cMyRLEnv()
            % Initialize observation settings
            ObservationInfo = rlNumericSpec([4 1]);
            ObservationInfo.Name = 'CartPole States';
            ObservationInfo.Description = 'x, dx, theta, dtheta';

            % Initialize action settings
            ActionInfo = rlFiniteSetSpec([-1 1]);
            ActionInfo.Name = 'CartPole Action';

            % The following line implements built-in functions of the RL environment
            obj = obj@rl.env.MATLABEnvironment(ObservationInfo,ActionInfo);

            % Initialize property values and precompute necessary values
            updateActionInfo(obj);
        end

        % Reset environment to initial state and return initial observation
        function InitialObservation = reset(obj)
            % Theta (+- .05 rad)
            T0 = 2 * 0.05 * rand - 0.05;
            % Thetadot
            Td0 = 0;
            % X
            X0 = 0;
            % Xdot
            Xd0 = 0;

            InitialObservation = [X0;Xd0;T0;Td0];
            obj.State = InitialObservation;

            % (Optional) Use notifyEnvUpdated to signal that the
            % environment is updated (for example, to update the visualization)
            notifyEnvUpdated(obj);
        end

        function [Observation,Reward,IsDone,Info] = step(obj,Action)
            Info = [];

            % Get action
            Force = getForce(obj,Action);

            % Unpack state vector
            XDot = obj.State(2);
            Theta = obj.State(3);
            ThetaDot = obj.State(4);

            % Cache to avoid recomputation
            CosTheta = cos(Theta);
            SinTheta = sin(Theta);
            SystemMass = obj.CartMass + obj.PoleMass;
            temp = (Force + obj.PoleMass*obj.HalfPoleLength*ThetaDot^2*SinTheta)...
                /SystemMass;

            % Apply motion equations
            ThetaDotDot = (obj.Gravity*SinTheta - CosTheta*temp)...
                / (obj.HalfPoleLength*(4.0/3.0 - obj.PoleMass*CosTheta*CosTheta/SystemMass));
            XDotDot  = temp - obj.PoleMass*obj.HalfPoleLength*ThetaDotDot*CosTheta/SystemMass;

            % Euler integration
            Observation = obj.State + obj.Ts.*[XDot;XDotDot;ThetaDot;ThetaDotDot];

            % Update system states
            obj.State = Observation;

            % Check terminal condition
            X = Observation(1);
            Theta = Observation(3);
            IsDone = abs(X) > obj.DisplacementThreshold || abs(Theta) > obj.AngleThreshold;
            obj.IsDone = IsDone;

            % Get reward
            Reward = getReward(obj);

            % (Optional) Use notifyEnvUpdated to signal that the
            % environment has been updated (for example, to update the visualization)
            notifyEnvUpdated(obj);
        end

    end

    methods

        function Reward = getReward(obj)
            if ~obj.IsDone
                Reward = obj.RewardForNotFalling;
            else
                Reward = obj.PenaltyForFalling;
            end
        end

        function Force = getForce(obj, Action)
            Force = Action;
        end

        function plot(obj)
            % Initiate the visualization
            obj.Figure = figure('Visible','on','HandleVisibility','off');
            ha = gca(obj.Figure);
            ha.XLimMode = 'manual';
            ha.YLimMode = 'manual';
            ha.XLim = [-3 3];
            ha.YLim = [-1 2];
            hold(ha,'on');
            % Update the visualization
            envUpdatedCallback(obj)
        end


    end

    properties(Access = protected)

        % Handle to figure
        Figure
    end

    methods(Access = protected)

        function envUpdatedCallback(obj)
            if ~isempty(obj.Figure) && isvalid(obj.Figure)
                % Set visualization figure as the current figure
                ha = gca(obj.Figure);

                % Extract the cart position and pole angle
                x = obj.State(1);
                theta = obj.State(3);

                cartplot = findobj(ha,'Tag','cartplot');
                poleplot = findobj(ha,'Tag','poleplot');
                if isempty(cartplot) || ~isvalid(cartplot) ...
                        || isempty(poleplot) || ~isvalid(poleplot)
                    % Initialize the cart plot
                    cartpoly = polyshape([-0.25 -0.25 0.25 0.25],[-0.125 0.125 0.125 -0.125]);
                    cartpoly = translate(cartpoly,[x 0]);
                    cartplot = plot(ha,cartpoly,'FaceColor',[0.8500 0.3250 0.0980]);
                    cartplot.Tag = 'cartplot';

                    % Initialize the pole plot
                    L = obj.HalfPoleLength*2;
                    polepoly = polyshape([-0.1 -0.1 0.1 0.1],[0 L L 0]);
                    polepoly = translate(polepoly,[x,0]);
                    polepoly = rotate(polepoly,rad2deg(theta),[x,0]);
                    poleplot = plot(ha,polepoly,'FaceColor',[0 0.4470 0.7410]);
                    poleplot.Tag = 'poleplot';
                else
                    cartpoly = cartplot.Shape;
                    polepoly = poleplot.Shape;
                end

                % Compute the new cart and pole position
                [cartposx,~] = centroid(cartpoly);
                [poleposx,poleposy] = centroid(polepoly);
                dx = x - cartposx;
                dtheta = theta - atan2(cartposx-poleposx,poleposy-0.25/2);
                cartpoly = translate(cartpoly,[dx,0]);
                polepoly = translate(polepoly,[dx,0]);
                polepoly = rotate(polepoly,rad2deg(dtheta),[x,0.25/2]);

                % Update the cart and pole positions on the plot
                cartplot.Shape = cartpoly;
                poleplot.Shape = polepoly;

                % Refresh rendering in the figure window
                drawnow();
            end
        end

        function updateActionInfo(obj)
            %this.ActionInfo.Elements = this.MaxForce*[-1 1];
        end

    end

end
