import numpy as np
from PIL import Image
from PIL import ImageDraw
import ImgNoiseGeneration as noise_gen
import torch
import os


class Pendulum:

    MAX_VELO_KEY = 'max_velo'
    MAX_TORQUE_KEY = 'max_torque'
    MASS_KEY = 'mass'
    LENGTH_KEY = 'length'
    GRAVITY_KEY = 'g'
    FRICTION_KEY = 'friction'
    DT_KEY = 'dt'
    SIM_DT_KEY = 'sim_dt'
    TRANSITION_NOISE_TRAIN_KEY = 'transition_noise_train'
    TRANSITION_NOISE_TEST_KEY = 'transition_noise_test'

    OBSERVATION_MODE_LINE = "line"
    OBSERVATION_MODE_BALL = "ball"

    def __init__(self,
                 img_size,
                 observation_mode,
                 generate_actions=False,
                 transition_noise_std=0.0,
                 observation_noise_std=0.0,
                 pendulum_params = None,
                 seed=0):

        assert observation_mode == Pendulum.OBSERVATION_MODE_BALL or observation_mode == Pendulum.OBSERVATION_MODE_LINE
        # Global Parameters
        self.state_dim = 2
        self.action_dim = 1
        self.img_size = img_size
        self.observation_dim = img_size ** 2
        self.observation_mode = observation_mode

        self.random = np.random.RandomState(seed)

        # image parameters
        self.img_size_internal = 128
        self.x0 = self.y0 = 64
        self.plt_length = 55 if self.observation_mode == Pendulum.OBSERVATION_MODE_LINE else 50
        self.plt_width = 8

        self.generate_actions = generate_actions

        # simulation parameters
        if pendulum_params is None:
            pendulum_params = self.pendulum_default_params()
        self.max_velo = pendulum_params[Pendulum.MAX_VELO_KEY]
        self.max_torque = pendulum_params[Pendulum.MAX_TORQUE_KEY]
        self.dt = pendulum_params[Pendulum.DT_KEY]
        self.mass = pendulum_params[Pendulum.MASS_KEY]
        self.length = pendulum_params[Pendulum.LENGTH_KEY]
        self.inertia = self.mass * self.length**2 / 3
        self.g = pendulum_params[Pendulum.GRAVITY_KEY]
        self.friction = pendulum_params[Pendulum.FRICTION_KEY]
        self.sim_dt = pendulum_params[Pendulum.SIM_DT_KEY]

        self.observation_noise_std = observation_noise_std
        self.transition_noise_std = transition_noise_std

        self.tranisition_covar_mat = np.diag(np.array([1e-8, self.transition_noise_std**2, 1e-8, 1e-8]))
        self.observation_covar_mat = np.diag([self.observation_noise_std**2, self.observation_noise_std**2])

    def sample_data_set(self, num_episodes, episode_length, full_targets, seed=None):
        """
        This function creates a number of sequences
        :param num_episodes: number of sequences
        :param episode_length: length of each sequence
        :param full_targets:
        :param seed:
        :return: imgs (image generated using noisy targets), targets (position of the penddulum in x,y),
                states (position and velocity in rad, rad/s), noisy targets (targets + observation noise)
        """
        if seed is not None:
            self.random.seed(seed)
        states = np.zeros((num_episodes, episode_length, self.state_dim))
        states[:, 0, :] = self._sample_init_state(num_episodes)

        continuous = np.zeros((1, episode_length * int(np.round(self.dt / self.sim_dt)), self.state_dim))
        continuous[:, 0, :] = self._sample_init_state(1)
        continuous, discrete = self._transition_function(continuous, actions=None, continuous_flag=1, episode_length=episode_length)

        for i in range(1, episode_length):
            states[:, i, :] = self._get_next_states(states[:, i - 1, :])
        #states -= np.pi #???????


        if self.observation_noise_std > 0.0:
            observation_noise = self.random.normal(loc=0.0, scale=self.observation_noise_std, size=states.shape)
            noise_discrete = self.random.normal(loc=0.0, scale=self.observation_noise_std, size=discrete.shape) #NOISE FOR THE EXAMPLE
        else:
            observation_noise = np.zeros(states.shape)
            noise_discrete = np.zeros(discrete.shape) #NOISE FOR THE EXAMPLE CASE

        cartesian = self.pendulum_kinematic(states)
        if self.observation_mode == Pendulum.OBSERVATION_MODE_LINE:
            noisy_discrete = discrete + noise_discrete #FOR EXAMPLE
            noisy_states = states + observation_noise
            noisy_cartesian = self.pendulum_kinematic(noisy_states)
        elif self.observation_mode == Pendulum.OBSERVATION_MODE_BALL:
            noisy_targets = cartesian + observation_noise
        imgs = self._generate_images(noisy_cartesian[..., :2])

        example = [continuous, discrete, noisy_discrete]

        #return imgs, cartesian[..., :(4 if full_targets else 2)], states, noisy_cartesian[..., :(4 if full_targets else 2)], example
        return imgs, states, noisy_states, example



    def _sample_init_state(self, nr_epochs):
        """
        Randomly initialize the states of the pendulum (theta, omega)
        :param nr_epochs: HERE IS THE NUMBER OF EPISODES AKA SEQUENCES
        :return: return an array of initial values of shape (#episodes, 2)
        """
        return np.concatenate((self.random.uniform(-0.5*np.pi, 0.5*np.pi, (nr_epochs, 1)), np.zeros((nr_epochs, 1))), 1)

    def _get_next_states(self, states):
        states = self._transition_function(states)
        """
        if self.transition_noise_std > 0.0:
            states[:, 1] += self.random.normal(loc=0.0,
                                               scale=self.transition_noise_std,
                                               size=[len(states)])
        """
        #NOISE ADDED ONLY TO VELOCITY
        #states[:, 0] = ((states[:, 0]) % (2 * np.pi))
        return states

    def _transition_function(self, states, continuous_flag=0, episode_length=None):
        nSteps = int(np.round(self.dt / self.sim_dt))

        if continuous_flag:
            for i in range(1, episode_length*nSteps):
                vel = states[:, i-1, 1] - self.g / self.length * np.sin(states[:, i-1, 0])*self.sim_dt \
                      + self.transition_noise_std * np.sqrt(self.sim_dt) * np.random.randn()
                states[:, i, 1] = vel
                states[:, i, 0] = states[:, i-1, 0] + self.sim_dt * vel
            discrete = states[:, ::nSteps, :]
            return states, discrete

        if nSteps != np.round(nSteps):
            print('Warning from Pendulum: dt does not match up')
            nSteps = np.round(nSteps)

        c = self.g * self.length * self.mass / self.inertia     # inertia = mass * self.length**2 / 3
        for i in range(0, int(nSteps)):
            velNew = states[..., 1:2] - self.g / self.length * np.sin(states[..., 0:1])*self.sim_dt + \
                     self.transition_noise_std * np.sqrt(self.sim_dt) * np.random.randn()
            states = np.concatenate((states[..., 0:1] + self.sim_dt * velNew, velNew), axis=1)
        return states

    def pendulum_kinematic(self, js_batch):
        """

        :param js_batch: STATES
        :return: array with position of pendulum in array frame (x,y) where x are rows and y are columns
        """
        theta = js_batch[..., :1]
        theta_dot = js_batch[..., 1:]
        x = np.sin(theta)
        y = np.cos(theta)
        x_dot = theta_dot * y
        y_dot = theta_dot * -x
        return np.concatenate([x, y, x_dot, y_dot], axis=-1)

    def _generate_images(self, ts_pos):
        imgs = np.zeros(shape=list(ts_pos.shape)[:-1] + [self.img_size, self.img_size], dtype=np.uint8)
        for seq_idx in range(ts_pos.shape[0]):
            print(seq_idx)
            for idx in range(ts_pos.shape[1]):
                imgs[seq_idx, idx] = self._generate_single_image(ts_pos[seq_idx, idx])

        return imgs

    @staticmethod
    def pendulum_default_params():
        return {
            Pendulum.MAX_VELO_KEY: 8,
            Pendulum.MAX_TORQUE_KEY: 10,
            Pendulum.MASS_KEY: 1,
            Pendulum.LENGTH_KEY: 1,
            Pendulum.GRAVITY_KEY: 9.81,
            Pendulum.FRICTION_KEY: 0,

            Pendulum.DT_KEY: 0.05,
            Pendulum.SIM_DT_KEY: 1e-4}

    def _transition_function_KNet(self, x): # KNet version, change states to x_t
        nSteps = self.dt / self.sim_dt

        if nSteps != np.round(nSteps):
            print('Warning from Pendulum: dt does not match up')
            nSteps = np.round(nSteps)

        c = self.g * self.length * self.mass / self.inertia
        for i in range(0, int(nSteps)):
            velNew = x[1:2] + self.sim_dt * (c * np.sin(x[0:1])                                                     
                                                     - x[1:2] * self.friction)
            x = torch.from_numpy(np.concatenate((x[0:1] + self.sim_dt * velNew, velNew), axis=0))
        return x

    def get_ukf_smothing(self, obs):
        batch_size, seq_length = obs.shape[:2]
        succ = np.zeros(batch_size, dtype=np.bool)
        means = np.zeros([batch_size, seq_length, 4])
        covars = np.zeros([batch_size, seq_length, 4, 4])
        fail_ct = 0
        for i in range(batch_size):
            if i % 10 == 0:
                print(i)
            try:
                means[i], covars[i] = self.ukf.filter(obs[i])
                succ[i] = True
            except:
                fail_ct +=1
        print(fail_ct / batch_size, "failed")

        return means[succ], covars[succ], succ

    def add_observation_noise(self, imgs, first_n_clean, r=0.2, t_ll=0.1, t_lu=0.4, t_ul=0.6, t_uu=0.9):
        return noise_gen.add_img_noise(imgs, first_n_clean, self.random, r, t_ll, t_lu, t_ul, t_uu)

    def _get_task_space_pos(self, joint_states):
        task_space_pos = np.zeros(list(joint_states.shape[:-1]) + [2])
        task_space_pos[..., 0] = np.sin(joint_states[..., 0]) * self.length
        task_space_pos[..., 1] = np.cos(joint_states[..., 0]) * self.length
        return task_space_pos

    def _generate_single_image(self, pos):
        x1 = pos[0] * (self.plt_length / self.length) + self.x0
        y1 = pos[1] * (self.plt_length / self.length) + self.y0
        img = Image.new('F', (self.img_size_internal, self.img_size_internal), 0.0)
        draw = ImageDraw.Draw(img)
        if self.observation_mode == Pendulum.OBSERVATION_MODE_LINE:
            draw.line([(self.x0, self.y0), (x1, y1)], fill=1.0, width=self.plt_width)
        elif self.observation_mode == Pendulum.OBSERVATION_MODE_BALL:
            x_l = x1 - self.plt_width
            x_u = x1 + self.plt_width
            y_l = y1 - self.plt_width
            y_u = y1 + self.plt_width
            draw.ellipse((x_l, y_l, x_u, y_u), fill=1.0)

        img = img.resize((self.img_size, self.img_size), resample=Image.ANTIALIAS)
        img_as_array = np.asarray(img)
        img_as_array = np.clip(img_as_array, 0, 1)
        return 255.0 * img_as_array

    def _kf_transition_function(self, state, noise):
        nSteps = self.dt / self.sim_dt

        if nSteps != np.round(nSteps):
            print('Warning from Pendulum: dt does not match up')
            nSteps = np.round(nSteps)

        c = self.g * self.length * self.mass / self.inertia
        for i in range(0, int(nSteps)):
            velNew = state[1] + self.sim_dt * (c * np.sin(state[0]) - state[1] * self.friction)
            state = np.array([state[0] + self.sim_dt * velNew, velNew])
        state[0] = state[0] % (2 * np.pi)
        state[1] = state[1] + noise[1]
        return state

    def pendulum_kinematic_single(self, js):
        theta, theat_dot = js
        x = np.sin(theta)
        y = np.cos(theta)
        x_dot = theat_dot *  y
        y_dot = theat_dot * -x
        return np.array([x, y, x_dot, y_dot]) * self.length


    def inverse_pendulum_kinematics(self, ts_batch):
        x = ts_batch[..., :1]
        y = ts_batch[..., 1:2]
        x_dot = ts_batch[..., 2:3]
        y_dot = ts_batch[..., 3:]
        val = x / y
        theta = np.arctan2(x, y)
        theta_dot_outer = 1 / (1 + val**2)
        theta_dot_inner = (x_dot * y - y_dot * x) / y**2
        return np.concatenate([theta, theta_dot_outer * theta_dot_inner], axis=-1)

    def _sample_action(self, shape):
        if self.generate_actions:
            return self.random.uniform(-self.max_torque, self.max_torque, shape)
        else:
            return np.zeros(shape=shape)

def data_gen():
    pend_params = Pendulum.pendulum_default_params()
    pend_params[Pendulum.FRICTION_KEY] = 0.1
    gdata = Pendulum(24, observation_mode=Pendulum.OBSERVATION_MODE_LINE,
                     transition_noise_std=0.1,
                     observation_noise_std=1e-5,
                     seed=0,
                     pendulum_params=pend_params)

    _, train_targets, train_states, noisy_trgt = generate_pendulum_filter_dataset(gdata, 500, 100, np.random.randint(2021))
    return train_targets, train_states, noisy_trgt


# cf. RKN code
def generate_pendulum_filter_dataset(pendulum, num_seqs, seq_length, seed):
    obs, targets, states, noisy_targets = pendulum.sample_data_set(num_seqs, seq_length, full_targets=False, seed=seed)
    obs, _ = pendulum.add_observation_noise(obs, first_n_clean=60, r=0.2, t_ll=0.0, t_lu=0.25, t_ul=0.75, t_uu=1.0)
    return obs, targets, states, noisy_targets


if __name__ == '__main__':
    img_size = 24
    train_seq_number = 3000
    train_seq_length = 150
    validation_seq_number = 100
    validation_seq_length = 150
    test_seq_number = 100
    test_seq_length = 150
    observation_noise_std = np.sqrt(0.00001)
    transition_noise_std = np.sqrt(0.000001)

    pend_params = Pendulum.pendulum_default_params()
    pend_params[Pendulum.FRICTION_KEY] = 0#0.1
    pend_params[Pendulum.SIM_DT_KEY] = 1e-5
    pend_params[Pendulum.DT_KEY] = 1e-2


    data = Pendulum(img_size=img_size,
                    observation_mode=Pendulum.OBSERVATION_MODE_LINE,
                    generate_actions=False,
                    transition_noise_std=transition_noise_std,
                    observation_noise_std=observation_noise_std,
                    pendulum_params=pend_params,
                    seed=0)
    # return imgs, states, noisy_states, example


    training_image, training_states, noisy_states, example = data.sample_data_set(num_episodes=train_seq_number,
                                                                                   episode_length=train_seq_length,
                                                                                   full_targets=False, seed=1)
    cv_image, cv_states, cv_noisy_states, _ = data.sample_data_set(num_episodes=validation_seq_number,
                                                                       episode_length=validation_seq_length,
                                                                       full_targets=False, seed=2)
    test_image, test_states, test_noisy_states, _ = data.sample_data_set(num_episodes=test_seq_number,
                                                                           episode_length=test_seq_length,
                                                                           full_targets=False, seed=3)

    #noisy_samples, factors = data.add_observation_noise(imgs, 0)

    #WRONG --> should use transpose instead of reshape
    #torch.save([training_input, training_target.reshape((training_target.shape[0],2,30)), cv_input, cv_target.reshape((cv_target.shape[0],2,30)), test_input, test_target.reshape((test_target.shape[0],2,40))], r".\Simulations\Pendulum\y24x24_Ttrain30_NE1000_NCV100_NT100_Ttest40_pendulum.pt")
    os.makedirs(r".\Datasets\Pendulum\v_-10dB", exist_ok=True)
    torch.save([[training_image, training_states, noisy_states],
                [cv_image, cv_states, cv_noisy_states], [test_image, test_states, test_noisy_states], example],  f"./Datasets/Pendulum/v_-10dB/pendulum_trans_var_{np.round(transition_noise_std**2, 4)}"
                                                                f"_obs_var_{np.round(observation_noise_std**2, 4)}_train_{train_seq_number}&{train_seq_length}" 
                                                                f"_val_{validation_seq_number}&{validation_seq_length}"
                                                                f"_test_{test_seq_number}&{test_seq_length}.pt")

    print("done")






