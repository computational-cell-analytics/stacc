
def main():
    config_path = args.config

    # Read config file
    with open(config_path) as file:
        parameters_dict = json.load(file)

    # Get parameters from config file
    patch_shape = tuple(parameters_dict["patch_shape"])
    batch_size = parameters_dict["batch_size"]
    num_workers = int(parameters_dict["num_workers"])
    my_iterations = parameters_dict["my_iterations"]
    alpha_loss = parameters_dict["alpha_loss"]
    checkpoint_path = parameters_dict["checkpoints"]
    model_name = parameters_dict["model_name"]
    learning_rate = parameters_dict["learning_rate"]
    pretrained = True if parameters_dict["pretrained"] == 1 else False
    json_dataset = parameters_dict["dataset"]
    epsilon = parameters_dict["eps"]
    sigma = parameters_dict["sigma"]
    lower_bound = parameters_dict["lower_bound"]
    upper_bound = parameters_dict["upper_bound"]
    
    # Read data path file
    with open(json_dataset) as dataset:
        dict_dataset = json.load(dataset)

        

    train_images, train_labels, val_images, val_labels, test_images, test_labels = split_dict_dataset(dict_dataset)
    
                   patch_shape: tuple, 
                   num_workers: int, 
    
    train_loader, val_loader, _ = StaccDataLoader(train_images, train_labels, val_images, val_labels, test_images, test_labels, 
                                                        patch_shape=patch_shape, num_workers=num_workers, batch_size=batch_size, 
                                                        sigma=sigma, lower_bound=lower_bound, upper_bound=upper_bound)
        
    return 


if __name__ == "__main__":
    main()