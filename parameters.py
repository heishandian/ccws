class Parameters:
    def __init__(self):
        self.output_filename = 'data/tmp'
        self.num_machine = 1000
        self.alpha = 0.3
        self.num_epochs = 5000  # number of training epochs  10000 -> 1
        self.simu_len = 10  # length of the busy cycle that repeats itself
        self.num_ex = 1  # number of sequences

        self.output_freq = 10  # interval for output and store parameters

        self.num_seq_per_batch = 10  # number of sequences to compute baseline
        self.episode_max_length = 200  # enforcing an artificial terminal

        self.num_res = 2  # number of resources in the system
        self.num_nw = 5  # maximum allowed number of work in the queue

        self.time_horizon = 20  # number of time steps in the graph
        self.max_job_len = 15  # maximum duration of new jobs
        self.res_slot = 10  # maximum number of available resource slots
        self.max_job_size = 10  # maximum resource request of new work
        self.f = 1000
        self.backlog_size = 60  # backlog queue size

        self.max_track_since_new = 10  # track how many time steps since last new jobs

        self.job_num_cap = 40  # maximum number of distinct colors in current work graph

        self.new_job_rate = 0.7  # lambda in new job arrival Poisson Process

        self.discount = 1  # discount factor

        # compact representation
        self.network_compact_dim = (self.num_res + 1) * \
                                   (self.time_horizon + self.num_nw) + 1  # + 1 for backlog indicator

        self.network_output_dim = self.num_nw + 1  # + 1 for void action

        self.delay_penalty = -1  # penalty for delaying things in the current work screen
        self.hold_penalty = -1  # penalty for holding things in the new work screen
        self.dismiss_penalty = -1  # penalty for missing a job because the queue is full

        self.num_frames = 1  # number of frames to combine and process
        self.lr_rate = 0.003  # learning rate
        self.rms_rho = 0.9  # for rms prop
        self.rms_eps = 1e-9  # for rms prop

        self.unseen = False  # change random seed to generate unseen example

        # supervised learning mimic policy

        self.w = [0.35, 0.35,
                  0.30]  # [w1, w2, w3] weight of resource feature, weight of reward feature, weight of time feature

        self.lmd = [0, 0.2, 0.4, 0.6, 0.8, 1]
        self.INITIAL_EPSILON = 0.5

    def compute_dependent_parameters(self):
        assert self.backlog_size % self.time_horizon == 0  # such that it can be converted into an image
        self.backlog_width = self.backlog_size / self.time_horizon
        self.network_input_height = self.time_horizon
        self.network_input_width = \
            (self.res_slot +
             self.max_job_size * self.num_nw) * self.num_res + \
            self.backlog_width + \
            1  # for extra info, 1) time since last new job

        # compact representation
        self.network_compact_dim = (self.num_res + 1) * \
                                   (self.time_horizon + self.num_nw) + 1  # + 1 for backlog indicator

        self.network_output_dim = self.num_nw + 1  # + 1 for void action
