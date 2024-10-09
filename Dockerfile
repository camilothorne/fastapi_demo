# Use an official Python runtime as a parent image
FROM python:3.10.14-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port streamlit runs on
EXPOSE 8000

# Command to run your Streamlit app
CMD ["fastapi", "run"]