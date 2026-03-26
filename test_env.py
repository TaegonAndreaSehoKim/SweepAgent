from env.grid_clean_env import GridCleanEnv

env = GridCleanEnv()
state = env.reset()

print("initial state:", state)
env.render()

for _ in range(5):
    action = env.sample_action()
    next_state, reward, done, info = env.step(action)

    print(f"\naction={action} ({env.ACTION_NAMES[action]})")
    print("next_state:", next_state)
    print("reward:", reward)
    print("done:", done)
    print("info:", info)

    env.render()

    if done:
        break