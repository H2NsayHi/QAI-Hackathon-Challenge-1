# STEP 1: Pull python image
FROM python:3.8

# STEP 2,3: CREATE WORK DIR AND COPY FILE TO WORK DIR
WORKDIR /clowns
COPY requirements.txt /clowns

# STEP 4,5,6: INSTALL NECESSARY PACKAGE
RUN pip install --upgrade pip && apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
RUN pip install -r requirements.txt
RUN pip install gdown

# STEP 7: Download file weight if needed
# RUN gdown "https://drive.google.com/uc?export=download&id=1VIplhJoaKPI08Qcdq6FhPk_dMvGOfDMp"

# STEP 8: RUN COMMAND
COPY . /clowns
CMD ["python", "./api.py"]
