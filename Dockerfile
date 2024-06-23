FROM python:3.10

COPY requirements.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Set the working directory
WORKDIR /app

EXPOSE 8501

# Set the entry point for the container
CMD ["streamlit", "run", "app.py"]