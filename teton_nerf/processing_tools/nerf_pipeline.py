import os
import click
from download_gdrive import GDriveDownloader
from nerf_process import NeRFProcess
import multiprocessing as mp
from multiprocessing import Pool
from detectron import SemanticSegmentor

class NeRFStudioPipeline:
    def __init__(self, train, name, options, global_env="~/miniconda3/envs/nerfstudio/bin/python"):
        self.downloader = GDriveDownloader()
        self.nerf_process = NeRFProcess(global_env)
        self.options = options
        self.train = train
        self.name = name
        self.Detectron = SemanticSegmentor()

    def process_and_train(self, local_zip_path, checkpoint_output, model):
        processed_output = "processed_data/" + os.path.basename(local_zip_path).strip(".zip")
        
        if not os.path.exists(processed_output):
            self.nerf_process.process_data(local_zip_path, processed_output)
        else:
            print(f"Processed data already exists for {os.path.basename(local_zip_path)}")

        if self.train:
            print(f"Training model {checkpoint_output}")
            try:
                self.nerf_process.train(processed_output, checkpoint_output, options=self.options, model=model)
            except Exception as e:
                print(f"Error: {e}")

    def train_raw_data(self, model):
        raw_data_path = "data/"
        dataset_names = [name for name in os.listdir(raw_data_path)]
        with Pool(4) as pool:  # Limit the number of processes to 8
            results = [pool.apply_async(self.process_and_train, args=(f"{raw_data_path}{name}", f"outputs/{name.strip('.zip')}", model)) for name in dataset_names]
            for result in results:
                result.get()
            pool.close()
            pool.join()

    def download_and_train(self, folder_id, model):
        folder_name, datasets = self.downloader.list_files_in_folder(folder_id)
        checkpoint_output = f"outputs/{folder_name}"
        with Pool(4) as pool:  # Limit the number of processes to 8
            for dataset in datasets:
                if dataset["mimeType"] in ["application/zip", "application/zip-x-compressed"]:
                    print(f"Processing dataset: {dataset['name']}")
                    folder_path = f"data/{folder_name}"
                    local_zip_path = os.path.join(folder_path, dataset["name"])
                    if not os.path.exists(local_zip_path):
                        local_zip_path = self.downloader.download_file(dataset["id"], dataset["name"], folder_path)
                    else:
                        print(f"Dataset {dataset['name']} already downloaded.")
                    pool.apply_async(self.process_and_train, args=(local_zip_path, checkpoint_output, model))
            pool.close()
            pool.join()
        print("All models trained successfully")
        return checkpoint_output

    def render_and_export(self, checkpoint_folder, export_mode, model, do_render):
        dataset_names = [name for name in os.listdir(checkpoint_folder) if os.path.isdir(os.path.join(checkpoint_folder, name))]
        checkpoints = [os.path.join(checkpoint_folder, name) for name in dataset_names]

        print("Found datasets:")
        print(dataset_names)
        
        configs = []
        for ckp in checkpoints:
            config = self.nerf_process.find_config(os.path.join(ckp, model))
            if config:
                configs.append(config)

        with Pool(4) as pool:
            for name, config in zip(dataset_names, configs):
                output_name = f"renders/{name}_{model}"
                output_dir = f"exports/{name}"
                
                if do_render and not os.path.exists(output_name):
                    pool.apply_async(self.nerf_process.render, args=(config, output_name))
                elif do_render:
                    print(f"Dataset already rendered for {name}")
                
                if export_mode is not None and not os.path.exists(output_dir):
                    print(f"Exporting to {output_dir}")
                    pool.apply_async(self.nerf_process.export, args=(config, export_mode, output_dir))
                elif export_mode is not None:
                    print(f"Data already exported for {name}")

            pool.close()
            pool.join()
            

@click.command()
@click.option("--model", default="semantic-depth-nerfacto", help="The type of model to train, see Nerfstudio docs for options")
@click.option("--folder_id", default=None, help="The id of the folder containing the dataset we want to train on.")
@click.option("--checkpoints", default=None, help="The folder containing model checkpoints.")
@click.option("--data", is_flag=True, help="If included processes and trains models on raw datasets located in data folder")
@click.option("--no-train", is_flag=True, help="If flag is added then don't train models after processing Polycam data")
@click.option("--render", is_flag=True, help="Output directory for the renderer videos. if nothing is specified no videos are rendered")
@click.option("--export", default=None, help="Export pointclouds from the trained models.")
@click.option("--name", default="", help="A name to append to the naming of models in order to distinguish runs with different settings")
@click.option("--option", multiple=True, help="A way to pass options to the config classes in Nerfstudio")
def main(model, folder_id, checkpoints, data, no_train, render, export, name, option):
    mp.set_start_method('spawn')
    train = not no_train
    print(f"Training set to {train}")
    options = " ".join(option)
    pipeline = NeRFStudioPipeline(train, name, options)
    if folder_id:
        checkpoint_folder = pipeline.download_and_train(folder_id, model)
    elif checkpoints:
        checkpoint_folder = checkpoints
    elif data:
        pipeline.train_raw_data(model)
    else:
        raise ValueError("Either an id for dataset download or a checkpoints folder must be provided.")
    
    if render or export:
        pipeline.render_and_export(checkpoint_folder, export, model, render)

if __name__ == "__main__":
    main()