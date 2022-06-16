import numpy as np
from PIL import Image
from PIL import ImageDraw
import torch
import os
import gc
from numpy.random import default_rng



class Pendulum:

    LENGTH_KEY = 'length'
    GRAVITY_KEY = 'g'
    SIMULATION_LENGTH_KEY = 'simulation_length'
    DT_KEY = 'dt'
    SIM_DT_KEY = 'sim_dt'
    TRANSITION_NOISE_TRAIN_KEY = 'transition_noise_train'
    TRANSITION_NOISE_TEST_KEY = 'transition_noise_test'
    rng = default_rng()

    def __init__(self,
                 img_size=24,
                 transition_noise_std=0.0,
                 observation_noise_std=0.0,
                 pendulum_params=None,
                 seed=0):
        self.state_dim = 2
        self.action_dim = 1
        self.img_size = img_size
        self.observation_dim = img_size ** 2
        self.random = np.random.RandomState(seed)

        # image parameters
        self.img_size_internal = 128
        self.x0 = self.y0 = 64
        self.plt_length = 55
        self.plt_width = 8

        # simulation parameters
        if pendulum_params is None:
            pendulum_params = self.pendulum_default_params()

        self.length = pendulum_params[Pendulum.LENGTH_KEY]
        self.g = pendulum_params[Pendulum.GRAVITY_KEY]

        self.simulation_length = pendulum_params[Pendulum.SIMULATION_LENGTH_KEY]
        self.sim_dt = pendulum_params[Pendulum.SIM_DT_KEY]
        self.dt = pendulum_params[Pendulum.DT_KEY]

        self.observation_noise_std = observation_noise_std
        self.transition_noise_std = transition_noise_std

    @staticmethod
    def pendulum_default_params():
        return {
            Pendulum.LENGTH_KEY: 1,
            Pendulum.GRAVITY_KEY: 9.81,

            Pendulum.SIMULATION_LENGTH_KEY: 2,
            Pendulum.DT_KEY: 0.005,
            Pendulum.SIM_DT_KEY: 1e-5}

    def sample_continuous_data_set(self, num_episodes, seed=None):
        """

        :param num_episodes: number of episodes/trajectories created
        :param seed: for reproducibility
        :return: a multidimensional array dim: (num_episodes, number_steps (simulation_length/sim_dt), 2 (position, velocity))
        """

        episode_length = int(np.round(self.simulation_length/self.sim_dt))

        if seed is not None:
            self.random.seed(seed)
        states = np.zeros((num_episodes, episode_length, self.state_dim))
        states[:, 0, :] = self._sample_init_state(num_episodes)

        for i in range(1, episode_length):
            states[:, i, :] = self._get_next_states(states[:, i - 1, :])

        return states

    def _sample_init_state(self, nr_epochs):
        """
        Randomly initialize the states of the pendulum (theta, omega), omega is always set to 0.
        :param nr_epochs: number of episodes/trajectories
        :return: return an array of initial values of shape (#episodes, 2)
        """
        return np.concatenate((self.rng.uniform(low=-0.5*np.pi, high=0.5*np.pi, size=(nr_epochs, 1)),
                               np.zeros((nr_epochs, 1))), 1)

    def _get_next_states(self, states):
        """
        Takes an array of states of dim: (num_episodes, 2) and using the discrete pendulum evolution equations computes
        the next states.
        :param states: array of previous states dim (num_episodes, 2)
        :return: array of next states dim (num_episodes, 2)
        """

        pos_new = states[..., 0:1] + self.sim_dt * states[..., 1:2]
        velNew = states[..., 1:2] - (self.g / self.length) * np.sin(states[..., 0:1]) * self.sim_dt + \
                 self.transition_noise_std * np.sqrt(self.sim_dt) * self.rng.standard_normal(states[..., 1:2].shape)

        states = np.concatenate((pos_new, velNew), axis=1)
        return states

    def decimate_data(self, states):
        """
        Takes as input the fine grained trajectory generated with sample_continuous_data_set, and subsamples it using
        taking one sample every dt/sim_dt samples.
        :param states: array of fine grained trajectories
        :return: array of coarse graind trajectory (decimated)
        """
        step = int(np.round(self.dt/self.sim_dt))
        return states[:, ::step, :]

    def add_observation_noise(self, states):
        """
        Adds gaussian noise on top of the observation
        :param states: array of trajectories
        :return: array of trajectories with observation noisy on top
        """
        states += self.observation_noise_std * self.rng.standard_normal(states.shape)
        return states

    def generate_images(self, states):
        cartesian = self.pendulum_kinematic(states)
        images = self._generate_images(cartesian)
        return images

    def pendulum_kinematic(self, js_batch):
        theta = js_batch[..., :1]
        x = np.sin(theta)
        y = np.cos(theta)

        return np.concatenate([x, y], axis=-1)

    def _generate_images(self, ts_pos):
        imgs = np.zeros(shape=list(ts_pos.shape)[:-1] + [self.img_size, self.img_size], dtype=np.uint8)
        for seq_idx in range(ts_pos.shape[0]):
            print(seq_idx)
            for idx in range(ts_pos.shape[1]):
                imgs[seq_idx, idx] = self._generate_single_image(ts_pos[seq_idx, idx])

        return imgs

    def _generate_single_image(self, pos):
        x1 = pos[0] * (self.plt_length / self.length) + self.x0
        y1 = pos[1] * (self.plt_length / self.length) + self.y0
        image = Image.new('F', (self.img_size_internal, self.img_size_internal), 0.0)
        draw = ImageDraw.Draw(image)
        draw.line([(self.x0, self.y0), (x1, y1)], fill=1.0, width=self.plt_width)
        image = image.resize((self.img_size, self.img_size), resample=Image.ANTIALIAS)
        img_as_array = np.asarray(image)
        img_as_array = np.clip(img_as_array, 0, 1)
        return 255.0 * img_as_array

if __name__ == "__main__":

    img_size = 24


    pend_params = Pendulum.pendulum_default_params()
    pend_params[Pendulum.SIM_DT_KEY] = 1e-5
    pend_params[Pendulum.DT_KEY] = 5e-3
    pend_params[Pendulum.SIMULATION_LENGTH_KEY] = 2
    data = Pendulum(img_size=img_size,
                    pendulum_params=pend_params,
                    seed=0)
    training_set_size = 3000
    validation_set_size = 1000
    test_set_size = 1000
    vs = [-20]
    r2s = [10, 2, 1, 0.5, 0.1, 0.01, 0.001, 0.0001]
    r2s = [0.0001]
    for r2 in r2s:
        for v in vs:
            q2 = np.round((r2 * 10**(int(v/10))), 7)
            print(f"Generating data for q2: {q2}.")
            data.transition_noise_std = np.sqrt(q2)

            continuous = data.sample_continuous_data_set(training_set_size + validation_set_size + test_set_size)
            os.makedirs(r"D:\Datasets\new_high_resolution_data/", exist_ok=True)
            np.savez(rf"D:\Datasets\new_high_resolution_data\pendulum_continuous_q2_{q2:.0e}_v_{v}.npz",
                     training_set=np.transpose(continuous[:training_set_size, :, :], axes=(0, 2, 1)),
                     validation_set=np.transpose(continuous[training_set_size:(training_set_size + validation_set_size), :, :],
                                                 axes=(0, 2, 1)),
                     test_set=np.transpose(continuous[training_set_size + validation_set_size:, :, :], axes=(0, 2, 1)),
                     q2=q2, sim_dt=data.sim_dt, dt=data.dt, simulation_length=data.simulation_length)

            decimated = data.decimate_data(continuous)
            os.makedirs(r".\Datasets\Pendulum\decimated_clean_data/", exist_ok=True)
            np.savez(rf".\Datasets\Pendulum\decimated_clean_data\pendulum_decimated_q2_{q2:.0e}_v_{v}.npz",
                     training_set=np.transpose(decimated[:training_set_size, :, :], axes=(0, 2, 1)),
                     validation_set=np.transpose(decimated[training_set_size:(training_set_size + validation_set_size), :, :],
                                                 axes=(0, 2, 1)),
                     test_set=np.transpose(decimated[(training_set_size + validation_set_size):, :, :], axes=(0, 2, 1)),
                     q2=q2, sim_dt=data.sim_dt, dt=data.dt, simulation_length=data.simulation_length)

            del continuous
            gc.collect()

            img = data.generate_images(decimated)
            os.makedirs(r".\Datasets\Pendulum\images_clean/", exist_ok=True)
            np.savez(rf".\Datasets\Pendulum\images_clean\pendulum_images_clean_q2_{q2:.0e}_v_{v}.npz",
                     training_set=img[:training_set_size, ...],
                     validation_set=img[training_set_size:(training_set_size + validation_set_size), ...],
                     test_set=img[(training_set_size + validation_set_size):, ...], q2=q2, sim_dt=data.sim_dt,
                     dt=data.dt, simulation_length=data.simulation_length)
            del img
            gc.collect()

            data.observation_noise_std = np.sqrt(r2)
            noisy = decimated + data.observation_noise_std * data.rng.standard_normal(decimated.shape)
            os.makedirs(r".\Datasets\Pendulum\decimated_noisy_data/", exist_ok=True)
            np.savez(rf".\Datasets\Pendulum\decimated_noisy_data\pendulum_decimated_noisy_q2_{q2:.0e}_r2_{r2:.0e}_v{v}.npz",
                     training_set=np.transpose(noisy[:training_set_size, :, :], axes=(0, 2, 1)),
                     validation_set=np.transpose(noisy[training_set_size:(training_set_size + validation_set_size), :, :],
                                                 axes=(0, 2, 1)),
                     test_set=np.transpose(noisy[(training_set_size + validation_set_size):, :, :], axes=(0, 2, 1)),
                     q2=q2, sim_dt=data.sim_dt, dt=data.dt, simulation_length=data.simulation_length, r2=r2, v=v)
            del noisy
            del decimated
            gc.collect()




