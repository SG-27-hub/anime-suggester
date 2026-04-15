# Use a Python image that already has some AI tools
FROM python:3.10

# Set up a new user named "user" with user ID 1000
RUN useradd -m -u 1000 user

# Switch to the "user" user
USER user

# Set home to the user's home directory
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Set the working directory to the user's home directory
WORKDIR $HOME/app

# Copy the requirements and install them
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Copy all your code and images with the correct permissions
COPY --chown=user . $HOME/app

# Start the app on Port 7860
CMD ["gunicorn", "-b", "0.0.0.0:7860", "--timeout", "120", "web:app"]# Use a Python image that already has some AI tools
FROM python:3.10

# Set the working directory inside the server
WORKDIR /code

# Copy your requirements and install them first (faster building)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Copy all your code and those 98 images
COPY . .

# Start the app on Port 7860 (Hugging Face's favorite port)
CMD ["gunicorn", "-b", "0.0.0.0:7860", "web:app"]
