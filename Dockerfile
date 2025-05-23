# Use an official Python base image
FROM python:3.10

# Set the working directory inside the container
WORKDIR /app

# Copy all app files into the container
COPY . .

# Upgrade pip
RUN pip install --upgrade pip

# Install system dependencies for audio processing
RUN apt-get update && apt-get install -y ffmpeg libsndfile1

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose Streamlit's default port
EXPOSE 8501

# Run Streamlit app
CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.enableCORS=false", "--server.enableXsrfProtection=false"]
