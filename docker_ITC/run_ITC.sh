# nvidia-docker run --rm -it -v /home/ytpc2019b/catkin_ws/src/ros_start/scripts:/home dockerfile:latest /bin/bash
xhost +local:user
    NV_GPU='6, 7' nvidia-docker run -it \
    --env=DISPLAY=$DISPLAY \
    --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
    --env="QT_X11_NO_MITSHM=1" \
    --rm \
    -p 8097:8097 \
    -v /home/akada/home/projects:/home \
    --net host \
    itc:latest \