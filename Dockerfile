# Use a Python image that already has some AI tools
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
