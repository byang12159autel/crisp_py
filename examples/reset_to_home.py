from crisp_py.robot import make_robot
import time

robot = make_robot("fr3")
robot.wait_until_ready()
robot.home()
time.sleep(1.0)
robot.shutdown()