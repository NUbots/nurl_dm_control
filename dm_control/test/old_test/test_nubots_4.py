from dm_control import viewer
from dm_control.suite import nubots

env = nubots.stand()

viewer.launch(env)
