import torch


def export_model(model_path, export_path):
    model = torch.load(model_path, map_location="cpu", weights_only=False)
    model_state = model["model_state"]
    model_kwargs = model["init"]["model_kwargs"]
    print(model_kwargs)
    print()
    print()
    torch.save(model_state, export_path)


def main():
    # export_model("./models/AGAR_best.pt", "./models/colonies.pt")
    export_model("./models/LiveCell_best.pt", "./models/cells.pt")


if __name__ == "__main__":
    main()
