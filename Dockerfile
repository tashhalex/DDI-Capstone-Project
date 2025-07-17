FROM continuumio/miniconda3

WORKDIR /app

COPY environment.yml .

RUN conda env update -f environment.yml

SHELL ["conda", "run", "-n", "ddi_capstone_env", "/bin/bash", "-c"]

COPY . .

EXPOSE 8501

CMD ["conda", "run", "-n", "ddi_capstone_env", "streamlit", "run", "app/streamlit_app.py"]