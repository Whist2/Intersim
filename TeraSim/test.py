from terasim import Simulator
from terasim.envs import EnvTemplate

sim = Simulator("examples/maps/Mcity/sim.sumocfg")
env = EnvTemplate()
sim.bind_env(env)

sim.start()
sim.run(steps=1000)
sim.close()
