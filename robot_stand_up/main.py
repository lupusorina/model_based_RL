import mujoco
import numpy as np
from mujoco import viewer

class Go1:
    def __init__(self, visualize=True):
        self.model = mujoco.MjModel.from_xml_path("xmls/scene_mjx_fullcollisions_flat_terrain.xml")
        self.data = mujoco.MjData(self.model)
        # Start in lying down position
        self.data.ctrl[:] = self.model.keyframe('pre_recovery').ctrl
        self.visualize = visualize
        if self.visualize:
            self.viewer = viewer.launch_passive(self.model, self.data)

    def step(self):
        mujoco.mj_step(self.model, self.data)
        if self.visualize:
            self.viewer.sync()
    
    def standup(self, dt):
        self.data.ctrl[:] =  self.model.keyframe('pre_recovery').ctrl * (1- dt) + \
                            self.model.keyframe('home').ctrl * dt 
        mujoco.mj_step(self.model, self.data)
        if self.visualize:
            self.viewer.sync()


def main():
    go1 = Go1()
    for i in range(300):
        go1.step()
    
    # not sure why causing it to go crazy
    dt = 0 
    tstep = 1000
    for i in range(tstep):
         print("standing")
         dt += 1/tstep
         go1.standup(dt)

    for i in range(10000):
        print("done")
        go1.step()
    

if __name__ == "__main__":
    main()
