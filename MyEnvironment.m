classdef MyEnvironment < rl.env.MATLABEnvironment

    %MYENVIRONMENT: Template for defining custom environment in MATLAB.

    %% Properties (set properties' attributes accordingly)
    properties
        % Specify and initialize environment's necessary properties

        % X and Y grid numbers
        XGrid = 10
        YGrid = 10

        % Grid size in meter (square grids)
        GridSize = 20.0
        Total_Grids = 100

        % Full dimension of the Grid
        XMax = 200
        YMax = 200

        % Max and Min Angle the agent can move in degree (in each step)
        MaxAngle = 0.9999
        MinAngle = 0

        % Sample time (S)
        Ts = 0.25

        % Max Distance the agent can travel in meter (in each sample time)
        MaxDistance = 5
        MinDistance = 0

        % System dynamics dont change for this interval in sec
        %FixDuration = 30

        SimuDuration = 25 % for now Simulation time is same as Fix duration i.e. no change in profit over time

        no_of_epochs = 100 % no. of steps possible in one episode


        % Coverage range (in m)--agent can cover upto this range.
        CovRange = 30



        % Penalty when the agent goes outside boundary
        PenaltyForGoingOutside = -1000

    end

    properties
        G = gridSetup() % to setup grid using function call and for loop.
    end


    properties
        % Initialize system state 4 values. All zeros.
        % They are -- [Collected Profit sum, Traveled distance, x, and y pos of the
        % agent]
        % [ProfitSum, DistTravel, agent_x, agent_y]'
        State = zeros(4,1)
    end

    properties(Access = protected)
        % Initialize internal flag to indicate episode termination
        IsDone = false
    end

    %% Necessary Methods
    methods
        % Contructor method creates an instance of the environment

        function this = MyEnvironment()
            % Initialize Observation settings
            ObservationInfo = rlNumericSpec([4 1]);
            ObservationInfo.Name = 'state';
            ObservationInfo.Description = 'Profit, Distance, x, y';

            % Initialize Action settings
            % Two actions -- distance and angle. Both limited by the
            % provided range
            ActionInfo = rlNumericSpec([2 1],'LowerLimit',[0;0],'UpperLimit',[1;1]);
            ActionInfo.Name = 'dist;angle';

            % The following line implements built-in functions of RL env
            this = this@rl.env.MATLABEnvironment(ObservationInfo,ActionInfo);


            % Initialize property values and pre-compute necessary values
            updateActionInfo(this);
        end

        % Apply system dynamics and simulates the environment with the
        % given action for one step.

        function [Observation,Reward,IsDone,LoggedSignals] = step(this,Action)

            LoggedSignals = [];

            % n is used to count the number of epochs taken in one episode.
            % when n == no_of_epochs in one episode then end the episode
            persistent n
            if isempty(n)
                 n = 1;
            else
                 n = n+1;
            end

            % Get actions
            %[dist,angle] = getMovement(this,Action);


            Force = getForce(this,Action);

            %Observation=this.State+Force;

            % Unpack state vector
            Profit = this.State(1);
            Distance = this.State(2);
            x = this.State(3);
            y = this.State(4);

            % Computation of the necessary values
            dist = Force(1)*5;
            angle = Force(2)*359.999;
            CosTheta = cosd(angle);
            SinTheta = sind(angle);
            x_new = x + dist*CosTheta;
            y_new = y + dist*SinTheta;

            % To compute the new profit after taking the actions

            % Idea is if the centre of a grid is within the coverage range
            % of the agent, then it is covered and its profit is obtained.
            P  = 0;
            for k = 1: this.Total_Grids
                if sqrt((x_new-this.G(k).X)^2 + (y_new-this.G(k).Y)^2)<= this.CovRange
                    P = P + this.G(k).Profit;
                end
            end

            new_Profit = P;
            dist_Traveled = dist;

            delta_profit = new_Profit-Profit;
            delta_dist = dist;

            % New Observation
            Observation = [new_Profit; dist_Traveled; x_new; y_new];

            % Update system states
            this.State = Observation;

            % Check the terminal condition
            IsDone = false; % need this to set initial value to this variable
            if n == this.no_of_epochs  % one episode has a fixed number of epochs. Once reached
                IsDone = true;
                this.IsDone = IsDone;     % make IsDone = TRUE
                n = 0;
            end

            % Reward::
            % If goes outside the region, penalize the agent
            if (x_new > this.XMax || y_new > this.YMax)
                penalty = this.PenaltyForGoingOutside;
            else
                penalty = 0;
            end
            Reward = 10*(delta_profit/delta_dist)+ penalty; %delta_dist is zero which leads to NaN reward

        end

        % Reset environment to initial state and output initial observation
        function InitialObservation = reset(this)
            % Profit sum goes to 0
            P0 = 0;
            % Distance travelled goes to 0
            D0 = 0;
            % Initial x pos of the robot
            X0 = 0;
            % Initial y pos of the robot
            Y0 = 0;

            InitialObservation = [P0;D0;X0;Y0];
            this.State = InitialObservation;
        end
    end

    methods
        % Helper methods to create the environment
        % Not sure how to update this two methods???
        %Answer: You don't need to use these two methods for every custom environment you create
        function force = getForce(this,action)
%             if ~ismember(action,this.ActionInfo.Elements)
%                 error('Action must be %g for going left and %g for going right.',-this.MaxForce,this.MaxForce);
%             end
            force = action;
        end
        % Update the action info based on Values
        % Not sure how to update this in my case??? kept same as cart pole
        function updateActionInfo(this)
            %this.ActionInfo.Elements = this.MaxForce*[-1 1];
        end



    end

end


function Grid = gridSetup() % sets up the grid properties

            % Grids' centre position and profit details
            % Grid is a struct which stores 3 values. x pos, y pos and profit
            % for each grid.
            % This information is necessary to compute the coverage profit sum
            Total_Grids = 100; %this.XGrid*this.YGrid;
            Grid(Total_Grids) = struct();
            G = 1;
            for i = 1:10
              for j = 1:10
                  Grid(G).X = 10+ (j-1)*(20); % x pos of each grid centre
                  Grid(G).Y = (10)+ (i-1)*(20); % y pos of each grid centre
                  Grid(G).Profit = round(10*rand());
                  G = G + 1;
              end
            end

end
