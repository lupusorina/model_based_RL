import mujoco
import numpy as np
from mujoco import viewer

class Go1:
    def __init__(self, visualize=True):
        self.model = mujoco.MjModel.from_xml_path("xmls/scene_mjx_fullcollisions_flat_terrain.xml")
        self.data = mujoco.MjData(self.model)
        self.visualize = visualize
        if self.visualize:
            self.viewer = viewer.launch(self.model, self.data)

    def step(self):
        mujoco.mj_step(self.model, self.data)
        if self.visualize:
            self.viewer.sync()

def main():
    go1 = Go1()
    for i in range(1000):
        go1.step()

if __name__ == "__main__":
    main()
