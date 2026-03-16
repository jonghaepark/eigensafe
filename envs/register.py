from gymnasium.envs.registration import register

def register_custom_envs():
    
    ## For EigenSafe

    register(
        id="Halfcheetah-run-low-v5",
        entry_point="envs.mujoco.half_cheetah_run_low_v5:HalfCheetahEnv",
        max_episode_steps=200,
    )

    register(
        id="Hopper-run-high-v5",
        entry_point="envs.mujoco.hopper_run_high_v5:HopperEnv",
        max_episode_steps=400,
    )

    register(
        id="Ant-ball-v5",
        entry_point="envs.mujoco.ant_goal_v5:AntGoalEnv",
        max_episode_steps=400,
    )

    register(
        id="LunarLander-safety",
        entry_point="envs.box2d.lunar_lander_safe:LunarLander",
        kwargs={"continuous": True},
        max_episode_steps=400,
        # reward_threshold=200,
    )
