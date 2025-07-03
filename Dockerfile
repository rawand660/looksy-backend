# Use an official Python image as a parent image. python:3.10-slim is a good lightweight choice.
FROM python:3.10-slim

# Set the working directory inside the container to /app
WORKDIR /app

# Install system-level dependencies required for building dlib and opencv-python
# build-essential contains C++ compilers (like g++), and cmake is required by dlib.
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libgtk2.0-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install the Python dependencies specified in requirements.txt
# --no-cache-dir is used to reduce the final image size by not storing the pip cache.
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application's source code (app.py, static/ folder, etc.) into the container at /app
COPY . .

# Expose the port that Render will provide via the $PORT environment variable (defaults to 10000)
# This line is good practice for documenting which port the container will listen on.
EXPOSE 10000

# Define the command to run when the container starts.
# We use Waitress, a pure-Python production-grade WSGI server.
# The 'shell' form (without []) is used so that the $PORT environment variable is correctly expanded by the shell.
CMD waitress-serve --host=0.0.0.0 --port=$PORT app:app