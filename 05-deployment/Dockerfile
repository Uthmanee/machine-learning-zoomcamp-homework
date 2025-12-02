FROM agrigorev/zoomcamp-model:2025

# create a directory in Docker named app and we're using it as work directory 
WORKDIR /app

# Copy pyproject.toml 
COPY ["pyproject.toml", "uv.lock", "./"]

# Copy any python files to the working directory of Docker 
COPY ["predict_docker.py",  "./"]
COPY ["predict_test.py",  "./"]

# Install uv and your dependencies
# Note that the command 'uv pip sync --system uv.lock' install necessary dependencies without creating a virtual envirnment. This 
# is the equivalent of the command 'RUN pipenv install --deploy --system' for pipenv'. 
# The flag '--system' install dependencies into the system-wide Python preventing a virtual environment from being created
RUN pip install .


# Expose the port
EXPOSE 9696

# Run the FastAPI application with uvicorn
ENTRYPOINT ["uvicorn", "predict_docker:app", "--host", "0.0.0.0", "--port", "9696"]