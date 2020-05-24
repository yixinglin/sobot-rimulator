## Using docker

The docker image can be built by executing

    docker build -t sobot-rimulator .

However, in order to run the GUI application, some additional effort is required, so that the docker dontainer may 
access the display of the host machine. 

### Mac

The following setup is based on the blog post https://cntnr.io/running-guis-with-docker-on-mac-os-x-a14df6a76efc, 
where you can find more detailed explanations.

Install `socat` and start listening:

    brew install socat
    socat TCP-LISTEN:6000,reuseaddr,fork UNIX-CLIENT:\"$DISPLAY\"

Open a new terminal and install `XQuartz`:

    brew install xquartz

Determine the IP address of your host machines network interface

    IP=$(ipconfig getifaddr en0)
    
Run the docker image by setting the `DISPLAY` environment variable to your IP address

    docker run -e DISPLAY=$IP:0 sobot-rimulator
    
### Linux (Ubuntu)

After building the image, the container must be allowed access to the display and can then be run

    $ xhost +local:root
    $ docker run -e DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix sobot-rimulator
