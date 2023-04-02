# Set base image (host OS)
FROM python:3.9-slim

# By default, listen on port 5000
EXPOSE 80/tcp

# Set the working directory in the container
WORKDIR /Data

# Copy the dependencies file to the working directory
COPY requirements.txt .

# Install any dependencies
RUN pip install -r requirements.txt

# Copy the content of the local src directory to the working directory
COPY . .

# Specify the command to run on container start
CMD ["gunicorn", "QUAAF_toolkit_flask_app:Flask_App", "-c", "./gunicorn.conf.py"]