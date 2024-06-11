
from app.client import supabase
import io
from app.services.ReadData import Dataset
import pandas as pd



def upload_file(file: io.BytesIO, filepath: str, content_type: str):
    try:
        supabase.storage.from_("user-data-files").update(
            path=filepath,
            file=file.getvalue(),
            file_options={"content-type": content_type, "upsert": "true"},
        )
    except Exception as e:
        supabase.storage.from_("user-data-files").upload(
            path=filepath,
            file=file.getvalue(),
            file_options={"content-type": content_type},
        )






def load_df(filename: str) -> Dataset:
    file = download(filename)
    df = pd.read_csv(io.StringIO(file))
    data = Dataset(df)
    return data


def download(filename: str, decode = True) -> str:
    file_bytes = supabase.storage.from_("user-data-files").download(filename)
    if decode: file_bytes = file_bytes.decode("utf-8")
    return file_bytes


def upload_csv(data: Dataset, filename: str) -> None:
    csv_buffer = io.BytesIO()
    data.df.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)
    supabase_filepath = f"user-id-1/project-id-1/data/{filename}"
    upload_file(csv_buffer, supabase_filepath, "text/csv")
