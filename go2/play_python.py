from unitree_sdk2py.core.channel import (
    ChannelPublisher,
    ChannelFactoryInitialize
)

import itertools

class Go2:
    def __init__(self):
        ChannelFactoryInitialize(0)
    
    def run(self):
        for i in itertools.count():
            pass


if __name__ == "__main__":
    robot = Go2()
    robot.run()
