## HACKATHON CHALLENGE 1 DOCKER

**Folder contains the modules**

- `extract_pairs.py`: This module used for loading model.
- `api.py`: This module used for expose API (keep as template)
- `Dockerfile`: Note: At step 7
  + Upload file_weight to google drive.
  + Set Editor permission for anyone have link.
  + Open [link](https://sites.google.com/site/gdocs2direct/) to create direct link. 

**Guideline for duilding docker-image and upload to s3**

- Step 1: Build docker image (Note: punc "." end of command): **docker build -t image_name .**
  + Example: *docker build -t khoinlg/aihkt_challenge_01:latest .*
- Step 2: Check docker image after building
  + *docker images*
- Step 3: Run created docker image: **docker run --name container_name -p <'local-port'>:5000 image_name**
  + Example: *docker run --name aihkt -p 5000:5000 khoinlg/aihkt_challenge_01:latest*
  + Note: If getting the error: 
  **docker: Error response from daemon: Conflict. The container name "/aihkt" is already in use by container "xxxxxx". You have to remove (or rename) that container to be able to reuse that name.**
   Please run **docker container rm -f xxx** to remove that container.
- Step 4: Test result by running the request.py: **python3 request.py**
- Step 5: After testing success, wrap docker to docker-image.tar: **docker save -o image_name.tar image_name:tag** (you can get info in step 2)
  + Example: *docker save -o aihkt_challenge_01.tar khoinlg/aihkt_challenge_01:latest*
- Step 6: Upload file docker image_name.tar to the submission page.
