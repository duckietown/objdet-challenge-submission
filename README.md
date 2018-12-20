# Object-Detection-Demo
This is the object detection repository for the duckietown demo.
We have trained several neural network models based on TensorFlow Object Detection API to detect duckies, Duckiebots, traffic lights, QR codes, 
intersection signs, stop signs, and (traffic) signal signs. For convenience, we provide a object detection docker image for you to use.
Check the instructions here to use it on the fly: http://docs.duckietown.org/DT18/opmanual_duckiebot/out/demo_objdet.html

<div class='requirements' markdown="1">

Requires: Initialized Duckiebots.

Requires: wheel calibration completed.[Wheel Calibration](#wheel-calibration)

Requires: Camera calibration completed.[Camera calibration](#camera-calib)

Requires: Joystick demo completed.[Joystick demo](#rc-control)

Requires: Duckietown with all classified objects.

Requires: Docker is installed on your computer.

Results: The duckiebot is able to detect objects using its camera.
</div>



## Images of expected results {#demo-objdet-expected}

<div figure-id="fig:objdet2" figure-caption="The expected result is bounding boxes around objects in duckietown.">
     <img src="objdet2.png" style='width: 30em'/>
</div>

<div figure-id="fig:objdet1" figure-caption="Another example, notice this one also has the duckiebot.">
     <img src="objdet1.png" style='width: 30em'/>
</div>

TODO: A high quality video will be uploaded before the deadline. For now these images are self explanatory of what is to be expected.

## Duckietown setup notes {#demo-objdet-Duckietown-setup}

The Duckietown used for this demo must have the following characteristics.

 * Several duckies placed on the road and at the side of the road.

 * Intersection sign, Stop sign and Signal sign.

 * QR codes on the ground and below the signs mentioned above.

 * Traffic lights at intersections

 * Duckiebots on the road.


 No cluttering of objects in one place. Allow enough space between each object. An example image is shown below.

## Duckiebot setup notes {#demo-objdet-Duckiebot-setup}


Put a duckie on top of the duckiebot.(Seriously)

## Pre-flight checklist {#demo-objdet-pre-flight}



The pre-flight checklist for this demo are:

**Check**: Battery level is sufficient.

**Check**: Docker Installed.

**Check**: Joystick container is running.

**Check**: Rospicam container is running.

**Check**: Base container is running.

**Check**: Put on your smartie pants.

## Demo instructions {#demo-objdet-run}



The following steps must be completed in order to run the object detector on your duckiebot.


**Step 1**: When the Duckiebot is powered on, make sure all the containers required are running. In your laptop, run

    laptop $ docker -H ![duckie_bot].local ps

to check whether the right containers are running or not. You can also check by going to the portainer webpage: `http://![duckie_bot].local:9000/#/containers`.

Note: If the required containers are running then skip to Step 4.

<div figure-id="fig:Containers" figure-caption="The containers that are required for this demo.">
     <img src="Containers.png" style='width: 30em'/>
</div>



**Step 2 (Optional)**: Launch the rospicam container so that we can capture images from the camera of Duckiebot.

    laptop $ docker -H ![duckie_bot].local run -it --name ros-picam --network host  --device /dev/vchiq -v /data:/data Duckietown/rpi-Duckiebot-ros-picam:master18


This command will run a container and create a ros node inside automatically. This ros node will publish the images captured by the camera to a ros topic.



**Step 3 (Optional)**: Launch the base container on the duckiebot.


    laptop $ docker -H ![duckie_bot].local run -it --network host --privileged --name base duckietown/rpi-duckiebot-base:master18 /bin/bash



**Step 4**: Launch the object detector container.

Note: For this command you need the Duckiebot's IP address. In order to obtain the Duckiebot IP address, you should ping your Duckiebot in another terminal and note down the IP address of your duckiebot.

    laptop $ docker run -it --name object_detection --network host -e ROS_MASTER_URI=http://![duckie_bot_IPaddress]:11311/  -e DUCKIEBOT_NAME=![duckie_bot]  -e ROS_HOSTNAME=![Name_Of_Your_Computer] zgxsin/object_detection:1.7

Notice that we have to set up the ROS_MASTER_URI variable so that the ros nodes can communicate with each other. This command will create a object_detection ros node automatically. It will listen to the camera image topic in step 2 and predict images and send the predicted images to another topic for visualization.



Note: You can replace the `1.7` in the above command with `1.6` to use a model with good accuracy but lower speed. There is a trade off between the two. Incase the version 1.7 is not working for you, please try version 1.6.



**Step 5**: In another terminal type:

    laptop $ dts start_gui_tools ![duckie_bot]

We can check whether everything is working inside this container. (See Tip 1)

After that, run the following command in the container

    container $ rqt_image_view

This will pop up a new GUI window. Select the `/!duckie_bot]/prediction_images` topic from the drop down menu.



**Step 6**: Move the Duckiebot using the joystick demo to different parts of Duckietown and see the magic. The live feed is a bit slow at the moment so please give it time to update.




## Troubleshooting and Tips {#demo-objdet-troubleshooting}


Symptom: The Duckiebot is not moving.


Resolution: Make sure that the joystick container is running. Note that the command for launching the joystick was changed to:

    laptop $ dts duckiebot keyboard_control ![duckie_bot]



Symptom: No images recorded.


Resolution: Make sure that the rospicam container is running.



Symptom: The ros nodes cannot communicate with each other.


Resolution: If you are using docker on Mac OSX, there seems to be an issue with the network of docker containers. We recommend to use docker on Ubuntu 16.04. We have tested it and everything is fine.
(Insert Image).



Symptom: The storage in the raspberry PI has reached its limit.


Resolution: Run `docker -H ![duckie_bot].local images` to whether dangling images exist and run `docker -H ![duckie_bot].local  rmi --force image_ID` to remove them.



Symptom: ERROR: unable to contact ROS master at [http://![Duckiebot_name].local:11311/]
The traceback for the exception was written to the log file.


Resolution: make sure your laptop and Duckiebot are on the same network.



Symptom: docker: Error response from daemon: Conflict. The container name "/object_detection" is already in use by container.


Resolution: Run the command  

      laptop $ docker container rm --force object_detection.

Repeat step 4.



**Tip 1**: We can check whether everything is working inside object detection container. Run `rosnode list` to check whether corresponding ros nodes are running. Run `rostopic list` to check whether the corresponding ros topics exist. You can run other ros-related command to do the check as well.

**Tip 2**: It is very important that the right containers are working. Double check to make sure.




## Demo failure demonstration {#demo-objdet-failure}

Failure is not an option sorry.
