import os
import subprocess
from multiprocessing import Pool
import click
from download_gdrive import GDriveDownloader


class NeRFProcess:
    def __init__(self, global_env):
        self.global_env = global_env

    def process_data(self, path: str, output: str):
        # process_path ="~/Desktop/Bachelorprojekt/nerfstudio/nerfstudio/scripts/process_data.py"
        process_path = "~/Desktop/Bachelorprojekt/nerfstudio/teton_nerf/teton_nerf/process_data/process_polycam.py"
        process = subprocess.run([f"{self.global_env} {process_path} polycam --data {path} --output_dir {output}"], shell=True)
        self.assert_process(process)
        return process

    def train(self, path: str, output: str, options: str, model: str):
        train_path = "~/Desktop/Bachelorprojekt/nerfstudio/nerfstudio/scripts/train.py"
        process = subprocess.run([f"{self.global_env} {train_path} {model} --data {path} --output-dir {output} --viewer.quit-on-train-completion True --logging.local-writer.enable False --vis wandb{' ' if options else ''}{options}"], shell=True)
        self.assert_process(process)
        return process

    def render(self, checkpoint: str, output_path: str):
        render_path = "~/Desktop/Bachelorprojekt/nerfstudio/nerfstudio/scripts/render.py"
        process = subprocess.run([f"{self.global_env} {render_path} dataset --load-config {checkpoint} --output-path {output_path} --split test"], shell=True)
        self.assert_process(process)
        return process
    
    def export(self, checkpoint: str, type: str, output_dir: str):
        export_path = "~/Desktop/Bachelorprojekt/nerfstudio/nerfstudio/scripts/exporter.py"
        if type == "pointcloud":
            process = subprocess.run([f"{self.global_env} {export_path} {type} --load-config {checkpoint} --output-dir {output_dir} --normal-method open3d"])
        else:
            process = subprocess.run([f"{self.global_env} {export_path} {type} --load-config {checkpoint} --output-dir {output_dir}"])
        self.assert_process(process)
        return process

    def find_config(self, checkpoint_path: str):
        for root, dirs, files in os.walk(checkpoint_path):
            for file in files:
                if file.endswith(".yml"):
                    return os.path.join(root, file)
        return None  # Return None if no config file is found
    
    def assert_process(self, process):
        if process.returncode != 0:
            # The subprocess failed; print the error
            print(f"Error: {process.stderr}")
        else:
            # The subprocess succeeded; print the output
            print(f"Output: {process.stdout}")


@click.command()
@click.option("--model", default="nerfacto", help="The type of model to train, see Nerfstudio docs for options")
@click.option("--id", help="The id of the folder containing the dataset we want to train on.")
def main(model, id):
    global_env = "~/miniconda3/envs/nerfstudio/bin/python"
    downloader = GDriveDownloader()
    nerf_process = NeRFProcess(global_env)
    folder_name, datasets = downloader.list_files_in_folder(id)
    checkpoint_output = f"outputs/{folder_name}"
    print(f"Training {len(datasets)} NeRFs")

    with Pool() as pool:
        for dataset in datasets:
            if dataset["mimeType"] == "application/zip" or dataset["mimeType"] == "application/zip-x-compressed":
                print(dataset["name"])
                folder_path = "data"
                local_zip_path = downloader.download_file(dataset["id"], dataset["name"], folder_path)
                pool.apply_async(nerf_process.process_and_train, args=(local_zip_path, checkpoint_output, model))
            else:
                continue

        pool.close()
        pool.join()

    print("All models trained successfully")

    checkpoint_path = checkpoint_output
    dataset_names = [dataset["name"].strip(".zip") for dataset in datasets if dataset["mimeType"] == "application/zip" or dataset["mimeType"] == "application/zip-x-compressed"]
    checkpoints = [os.path.join(checkpoint_path, name) for name in dataset_names]

    configs = []
    for ckp in checkpoints:
        configs.append(nerf_process.find_config(ckp + f"/{model}"))

    with Pool() as pool:
        for name, config in zip(dataset_names, configs):
            output = f"renders/{name}.mp4"
            pool.apply_async(nerf_process.render, args=(config, output))

        pool.close()
        pool.join()

    print("All videos rendered successfully")


if __name__ == "__main__":
    main()