cmake_minimum_required(VERSION 3.0.2)
project(fly_drone)

find_package(catkin REQUIRED COMPONENTS
  rospy
)


include_directories(
  ${catkin_INCLUDE_DIRS}
)

