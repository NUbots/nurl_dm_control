import numpy as np
from dm_control.locomotion import soccer as dm_soccer


"""Interactive viewer for MuJoCo soccer environment."""

import functools
from absl import app
from absl import flags
from dm_control.locomotion import soccer
from dm_control import viewer

FLAGS = flags.FLAGS

flags.DEFINE_enum("walker_type", "HUMANOID", ["BOXHEAD", "ANT", "HUMANOID"],
                  "The type of walker to explore with.")
flags.DEFINE_bool(
    "enable_field_box", True,
    "If `True`, enable physical bounding box enclosing the ball"
    " (but not the players).")
flags.DEFINE_bool("disable_walker_contacts", False,
                  "If `True`, disable walker-walker contacts.")
flags.DEFINE_bool(
    "terminate_on_goal", False,
    "If `True`, the episode terminates upon a goal being scored.")



# Instantiates a 2-vs-2 BOXHEAD soccer environment with episodes of 10 seconds
# each. Upon scoring, the environment reset player positions and the episode
# continues. In this example, players can physically block each other and the
# ball is trapped within an invisible box encapsulating the field.
environment_loader=functools.partial(
            soccer.load,
            team_size=2,
            walker_type=soccer.WalkerType[FLAGS.walker_type],
            disable_walker_contacts=FLAGS.disable_walker_contacts,
            enable_field_box=FLAGS.enable_field_box,
            keep_aspect_ratio=True,
            terminate_on_goal=FLAGS.terminate_on_goal)

# Retrieves action_specs for all 4 players.
action_specs = environment_loader.action_spec()

timestep = environment_loader.reset()


def main(argv):
#   if len(argv) > 1:
#     raise app.UsageError("Too many command-line arguments.")
  
    viewer.launch(environment_loader=environment_loader)

    while not timestep.last():
        actions = []
        for action_spec in action_specs:
            action = np.random.uniform(
                action_spec.minimum, action_spec.maximum, size=action_spec.shape)
            actions.append(action)
        timestep = environment_loader.step(actions)

        for i in range(len(action_specs)):
            print(
                "Player {}: reward = {}, discount = {}, observations = {}.".format(
                    i, timestep.reward[i], timestep.discount, timestep.observation[i]))
            
if __name__ == "__main__":
  app.run(main)
