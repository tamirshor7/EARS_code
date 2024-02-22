import os
import datetime

SUMMARY_FILE_NAME: str = 'summary.txt'

def get_listed_files(path, summary_file_name=SUMMARY_FILE_NAME, summary_file_path=None):
    if summary_file_path is None:
        summary_path = os.path.join(path, summary_file_name)
    else:
        os.makedirs(summary_file_path, exist_ok=True)
        summary_path = os.path.join(summary_file_path, summary_file_name)
    if os.path.exists(summary_path):
        last_modified_time = datetime.datetime.fromtimestamp(os.path.getmtime(path))
        with open(summary_path, "r") as f:
            last_saved_modified_time = datetime.datetime.fromtimestamp(float(f.readline()))
            elapsed_time = last_modified_time-last_saved_modified_time
            if elapsed_time < datetime.timedelta(minutes=3):
                listed_files = f.readlines()
                listed_files = list(map(lambda x:x.removesuffix('\n'), listed_files))
                return listed_files
        try:
            os.remove(summary_path)
        except Exception as e:
            print(f"Couldn't remove {summary_path}. Got {e}")

    listed_files = generate_summary(path, summary_path)
    return listed_files

def generate_summary(path, summary_path):
    last_modified_time = f"{os.path.getmtime(path)}\n"
    listed_files = os.listdir(path)
    file_names = [f"{p}\n" for p in listed_files]
    with open(summary_path, "w") as f:
        f.write(last_modified_time)
        f.writelines(file_names)
    return listed_files