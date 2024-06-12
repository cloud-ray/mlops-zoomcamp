# Use the provided base image
FROM agrigorev/zoomcamp-model:mlops-2024-3.10.13-slim

# Set the working directory
WORKDIR /app

# Copy the script to the container
COPY homework-04.py .

# Install required libraries
RUN pip install pandas scikit-learn pyarrow fastparquet

# Command to run the script
CMD ["python", "homework-04.py", "--year", "2023", "--month", "5"]
