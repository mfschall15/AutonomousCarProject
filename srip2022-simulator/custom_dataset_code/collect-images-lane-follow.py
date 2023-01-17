import os
import cv2
import time
from resources.gyminterface import GymInterface
from datetime import datetime
import pandas as pd
import csv

''' Class to drive the virtual car '''
class VirtualDriver():

  __image_dir = ""
  __sub_img_dir = 0
  __image = None
  __gym = None
  __x = 0
  __y = 0
  __fps = 0
  __sample_time = 0
  __driving = False
  __image_count = 0
  

  def __init__(self, image_dir, sim_config, fps):
    # Config the simulator
    self.__image_dir = image_dir
    self.__gym = GymInterface(gym_config=sim_config)
    time.sleep(1)
    self.__gym.step(0.0, 0.0, 0.0, True)
    self.__fps = fps
    self.__csv = image_dir + '/' + '/data.csv'

    # Create data directory
    if not os.path.isdir(image_dir):
      
      os.makedirs(image_dir)
      print(f"Directory '{image_dir}' created successfully.")
    
    
    with open(self.__csv, 'w', newline = '') as csvfile:
        filewriter = csv.writer(csvfile, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)
        filewriter.writerow(['trial', 'x_pos', 'y_pos'])       
    
    return

  def drive(self):
    # Set up callback
    cv2.namedWindow("Controller")
    cv2.setMouseCallback("Controller", self.__mouse_callback)

    # Render initial frame to show the window
    step = self.__gym.step(0.0, 0.0, 0.0, False)
    self.__image = cv2.resize(cv2.cvtColor(step[0], cv2.COLOR_RGB2BGR), (224, 224))


    # self.__image = cv2.cvtColor(step[0], cv2.COLOR_RGB2BGR)
    cv2.imshow("Controller", self.__image)
    cv2.waitKey(1)

    # Enter into main loop until user exits
    try:
      while True:
        step = self.__gym.step(self.__x, self.__y, 0.0, False)
        self.__image = cv2.resize(cv2.cvtColor(step[0], cv2.COLOR_RGB2BGR), (224, 224))

        if self.__driving:
            if (time.time() - self.__sample_time > self.__fps):
                self.__sample_time = time.time()
                self.__image_count = self.__image_count + 1 if self.__image_count < 20 else 1
                img_name = f"{self.__image_count}.jpg"
                if self.__image_count == 1:
                    self.__sub_img_dir += 1
                    print(self.__sub_img_dir)


                cv2.imwrite(self.__image_dir + "/" + f"{self.__sub_img_dir}" + "_" + img_name, self.__image)
                with open(self.__csv, 'a', newline='') as csvfile:
                    filewriter = csv.writer(csvfile, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)
                    filewriter.writerow([f"{self.__sub_img_dir}" + "_" + img_name, self.__x, self.__y]) 

        # self.__image = cv2.cvtColor(step[0], cv2.COLOR_RGB2BGR)
        cv2.imshow("Controller", self.__image)
        k = cv2.waitKey(10)
        if k == ord('r'):
          self.__driving = False
          self.__gym.step(0.0, 0.0, 0.0, True)
          time.sleep(1)
        elif k == ord('q'):
          self.__cleanup()
          exit()
    except KeyboardInterrupt:
      car.__cleanup()
    return

  def __mouse_callback(self, event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONUP:
      self.__driving = False
      self.__x = 0
      self.__y = 0
      return

    if event == cv2.EVENT_LBUTTONDOWN:
      self.__driving = True

    if self.__driving:
      self.__x =  2.0 * (int(x) / 224.0 - 0.5)
      self.__y = -2.0 * (int(y) / 224.0 - 0.8)
      # self.__x =  2.0 * (int(x) / 224.0 - 0.5)
      # self.__y = -2.0 * (int(y) / 224.0 - 0.8)

      
    return

  def __cleanup(self):
    cv2.destroyAllWindows()
    self.__gym.onShutdown()
    print("\nExiting...")
    return


''' Main apllication entrypoint '''
if __name__ == '__main__':

  print("\n******************** Getting Started ********************")
  directory = input("Specify the name of the directory where images will be stored: ")
  fps = 1.0/float(input("Specify how frequently you want to save images (in FPS): "))
  ip_address = input("Input the IP addess of donkey_sim (you can check ipconfig or ifconfig): ")
  print("***********************************************************\n")

  # Gym configuration
  config = {
    'car': {
      # Vehicle Settings
      'racer_name': 'Marcus S',            # Your name
      'car_name'  : 'Yeet Mobile',              # displays above car
      'bio'       : '...',                 # Description of your vehicle
      'country'   : 'US',                  # Your country
      "guid"      : "GO_TRITON_AI",        # Your GUID
      'body_style': 'car01',               # The vehicle style – options include: 'car01', 'f1', or 'donkey'
      'body_rgb'  : (24, 43, 200),         # Color of the vehicle as (R,G,B)
      'font_size' : 50,                    # The size of the text

      # The Simulated Car Camera Settings
      "fov"       : 80,                    # The field-of-view of the camera
      "img_w"     : 224,                   # The image width
      "img_h"     : 224,                   # The image height
      "img_d"     : 3,                     # The image channels (R,G,B)
      "img_enc"   : 'JPG',                 # The format to save images
      "offset_x"  : 0.0,                   # The horizontal offset of the camera
      "offset_y"  : 2.0,                   # The vertical offset of the camera
      "offset_z"  : 0.0,                   # The forward offset of the camera
      "rot_x"     : 20.0,                  # The horizontal tilt of the camera
      "rot_y"     : 0,                     # The vertical tilt of the camera
      "fish_eye_x": 0.0,                   # The horizontal fisheye of the camera
      "fish_eye_y": 0.0,                   # The vertical fisheye of the camera
    },

    'default_connection': 'local',         # The connection type – 'local' or 'remote'

    'local_connection': {
      'host': ip_address,                  # The host computer IP address (e.g., '192.168.86.116')
      'port': 9091,                        # The server port running the simulator (9091 by default)
      'artificial_latency': 0,             # Use to add simulated network latency
      'scene_name': 'warren',              # The virtual racetrack – options include:
                                           #   'warren', 'thunderhill', 'mini_monaco', 'roboracingleague_1',
                                           #   'generated_track', 'generated_road', 'warehouse', 'sparkfun_avc', 'waveshare'
    },

    'remote_connection': {
      'host': 'donkey-sim.roboticist.dev', # The host computer IP address (e.g., 'donkey-sim.roboticist.dev')
      'port': 9091,                        # The server port running the simulator (9091 by default)
      'artificial_latency': 0,             # Use to add simulated network latency
      'scene_name': 'warren',              # The virtual racetrack – options include:
                                           #   'warren', 'thunderhill', 'mini_monaco', 'roboracingleague_1',
                                           #   'generated_track', 'generated_road', 'warehouse', 'sparkfun_avc', 'waveshare'
    },

    'lidar': {
      'enabled': False,                  # Enable or disable a virtual lidar
      'deg_inc': 1,                      # The degree increment between each ray of the lidar
      'max_range': 30.0,                # The maximum range of the lidar laser
    }
  }

  print ("\n************************ Controls ************************")
  print("\nHold down the mouse on the \"Controller\" window to drive.")
  print("Click farther out to drive faster or closer to drive slower.")
  print("Note: if you experience a timeout, please check your IP.")
  print("\n***********************************************************\n")
  car = VirtualDriver(image_dir=directory, sim_config=config, fps=fps)
  car.drive()
