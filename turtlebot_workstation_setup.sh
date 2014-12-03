#!/bin/bash
# INSTALL ROS
sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu trusty main" > /etc/apt/sources.list.d/ros-latest.list' 
wget https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -O - | sudo apt-key add - 
sudo apt-get update 
sudo apt-get install ros-indigo-desktop-full 
apt-cache search ros-indigo 
sudo rosdep init 
rosdep update 
echo "source /opt/ros/indigo/setup.bash" >> ~/.bashrc 
source ~/.bashrc 
sudo apt-get install python-rosinstall 
# END INSTALL ROS

# INSTALL SYSTEM DEPENDENCIES
sudo apt-get install python-rosdep python-wstool ros-indigo-ros 
sudo rosdep init 
rosdep update

# INSTALL ROCON
mkdir ~/rocon 
cd ~/rocon 
wstool init -j5 src https://raw.github.com/robotics-in-concert/rocon/indigo/rocon.rosinstall 
source /opt/ros/indigo/setup.bash 
rosdep install --from-paths src -i -y 
catkin_make

# INSTALL KOBUKI
mkdir ~/turtlebot 
cd ~/turtlebot 
wstool init src -j5 https://raw.github.com/yujinrobot/yujin_tools/master/rosinstalls/indigo/turtlebot.rosinstall 
source ~/kobuki/devel/setup.bash 
rosdep install --from-paths src -i -y 
catkin_make 
source ~/turtlebot/devel/setup.bash

# INSTALL TURTLEBOT
mkdir ~/turtlebot 
cd ~/turtlebot 
wstool init src -j5 https://raw.github.com/yujinrobot/yujin_tools/master/rosinstalls/indigo/turtlebot.rosinstall 
source ~/kobuki/devel/setup.bash 
rosdep install --from-paths src -i -y 
catkin_make -DCATKIN_BLACKLIST_PACKAGES="map_store" 
source ~/turtlebot/devel/setup.bash 

# FOR DETECTION OF KOBUKI BASE
rosrun kobuki_ftdi create_udev_rules 
echo "source ~/turtlebot/devel/setup.bash" >> ~/.bashrc

# SYNCHRONIZE CLOCKS
sudo apt-get install chrony 
sudo ntpdate ntp.ubuntu.com




