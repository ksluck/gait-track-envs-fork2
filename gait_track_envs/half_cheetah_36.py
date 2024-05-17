import gym
import numpy as np
from gym import utils

from .jinja_mujoco_env import MujocoEnv


class HalfCheetahEnv(MujocoEnv, utils.EzPickle):
    def __init__(self, parametric=True, init_task=None):
        self.original_lengths = np.array([ .145, .15, .094, .133, .106, .07])
        self.current_lengths = np.array(self.original_lengths)
        self.model_args = {"size": list(self.original_lengths)}

        self.markers = ["thigh", "shin", "foot", "foottip"]
        self.legs = ["b", "f"]
        self.origin = "torso"

        if parametric:
            MujocoEnv.__init__(self, 'parametric_half_cheetah_36.xml', 5)
        else:
            MujocoEnv.__init__(self, 'half_cheetah.xml', 5)
        utils.EzPickle.__init__(self)

        #self.min_task = np.ones_like(self.original_lengths)*0.035
        self.min_task = self.original_lengths*0.5
        self.max_task = self.original_lengths*2.0

        self.parametric = parametric

        if init_task:
            task = self.get_test_tasks()[init_task]
            self.set_task(*task)

    def get_test_tasks(self):
        return {"normal": np.array( [*self.original_lengths] ),
                "short": np.array( [*(self.original_lengths*0.5)] ),
                "long": np.array( [*(self.original_lengths*2)] )}

    def set_random_task(self):
        self.set_task(*self.sample_task())

    def sample_task(self):
        return np.random.uniform(self.min_task, self.max_task, self.min_task.shape)

    def sample_tasks(self, num_tasks=1):
        return np.stack([self.sample_task() for _ in range(num_tasks)])

    def get_task(self):
        return np.copy(self.current_lengths)

    @property
    def limb_segment_lengths(self):
        return self.current_lengths.reshape(2, 3)*2
    
    @property
    def morpho_params(self):
        assert self.current_lengths.flatten().shape == (6, )
        return self.current_lengths.flatten()

    def set_task(self, *task):
        if not self.parametric:
            raise TypeError("Attempting to modify a non-parametric Cheetah!")
        if len(task) == len(self.current_lengths):
            self.current_lengths[:] = task
        else:
            raise ValueError("Incorrect task shape")
        self.model_args = {"size": list(self.current_lengths)}
        self.build_model()
        
    def reset(self):
        obs, _ = MujocoEnv.reset(self)
        obs[0] = obs[0] / self.init_height
        return obs, {}
        

    def step(self, action):
        xposbefore = self.sim.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        xposafter = self.sim.data.qpos[0]
        ob = self._get_obs()
        reward_ctrl = - 0.1 * np.square(action).sum()
        reward_run = 1.25 * (xposafter - xposbefore)/self.dt
        reward = np.amax([(reward_ctrl + reward_run), 0.0])
        terminated = False
        truncated = False

        # Get pos/vel of the feet
        track_info = self.get_track_dict()


        info = {"reward_run": reward_run, "reward_ctrl": reward_ctrl, 'reward_sum':reward,
                **track_info}
        return ob, reward, terminated, truncated, info

    def _get_obs(self):
        qpos = self.sim.data.qpos.flat[1:]
        qvel = self.sim.data.qvel.flat
        #qpos[0] -= self.init_height
        qpos[0] = qpos[0] #/ self.init_height

        return np.concatenate([
            qpos,
            qvel,
        ])

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(low=-.1, high=.1, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.standard_normal(self.model.nv) * .1
        qpos[1] += 2
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5

    def set_sim_state(self, state):
        return self.sim.set_state(state)

    def get_sim_state(self):
        return self.sim.get_state()


def register_half_cheetah(env_name):
    if env_name == "GaitTrackHalfCheetah36-v0":
        kwargs = {"parametric": True}
    elif env_name == "GaitTrackHalfCheetah36Original-v0":
        kwargs = {"parametric": False}
    else:
        raise ValueError("Unknown env name")

    gym.envs.register(
            id=env_name,
            entry_point="%s:HalfCheetahEnv" % __name__,
            max_episode_steps=1000,
            kwargs=kwargs,
    )
