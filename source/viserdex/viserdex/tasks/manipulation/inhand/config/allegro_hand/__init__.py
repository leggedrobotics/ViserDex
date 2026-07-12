import gymnasium as gym

from . import allegro_env_cfg

from . import agents


gym.register(
    id="Isaac-ViserDex-Repose-Allegro-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.allegro_env_cfg:ViserDexAllegroEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:SimpleAllegroPPORunnerCfg",
    },
)


gym.register(
    id="Isaac-ViserDex-Repose-Allegro-Estimator-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.allegro_env_cfg:ViserDexAllegroPoseEstimatorEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:SimpleAllegroPPORunnerCfg",
    },
)


gym.register(
    id="Isaac-ViserDex-Repose-Allegro-Estimator-Eval-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.allegro_env_cfg:ViserDexAllegroPoseEstimatorEvalEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:SimpleAllegroPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-ViserDex-Repose-Allegro-Estimator-Camera-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.allegro_env_cfg:ViserDexAllegroPoseEstimatorCameraEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:SimpleAllegroPPORunnerCfg",
    },
)


gym.register(
    id="Isaac-ViserDex-Repose-Allegro-Estimator-NoAug-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.allegro_env_cfg:ViserDexAllegroPoseEstimatorUnaugmentedEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:SimpleAllegroPPORunnerCfg",
    },
)


gym.register(
    id="Isaac-ViserDex-Repose-Allegro-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.allegro_env_cfg:ViserDexAllegroEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:SimpleAllegroPPORunnerCfg",
    },
)
